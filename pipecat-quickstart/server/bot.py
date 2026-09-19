#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-quickstart - Pipecat Voice Agent

This bot uses a cascade pipeline: Speech-to-Text → LLM → Text-to-Speech

Required AI services:
- Self-hosted SenseVoice (Speech-to-Text)
- Mify (LLM)
- Self-hosted TTS (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import sys
import httpx
import subprocess
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, Frame, LLMRunFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner
from pipecat.frames.frames import Language

load_dotenv(override=True)


class CustomOpenAITTSService(OpenAITTSService):
    """
    Override OpenAITTSService to completely bypass voice validation and handle decoding MP3 -> PCM
    """
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """
        Generate TTS audio from text. Overridden to remove the VALID_VOICES check,
        and request MP3 format to decode it cleanly to Raw PCM via FFmpeg, since the TTS 
        server seems to return MP3 even when asked for PCM.
        """
        voice = self._settings.voice
        if voice is None:
            yield ErrorFrame(error="OpenAI TTS voice must be specified")
            return
            
        try:
            # Request MP3 explicitly, as red:5050 seems to return MP3 encoding even if pcm is asked
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": voice,
                "response_format": "mp3", 
            }
            if self._settings.speed is not None:
                create_params["speed"] = self._settings.speed

            # Start FFmpeg subprocess to stream-decode MP3 -> 16kHz Mono 16-bit PCM (Pipecat standard)
            sample_rate = self.sample_rate or 16000 # Pipecat default audio is typically 16000 for webrtc
            ffmpeg_cmd = [
                "ffmpeg", "-i", "pipe:0", 
                "-f", "s16le", 
                "-acodec", "pcm_s16le",
                "-ac", "1", 
                "-ar", str(sample_rate), 
                "pipe:1"
            ]
            
            import asyncio
            proc = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )

            async def read_stdout():
                while True:
                    chunk = await proc.stdout.read(self.chunk_size)
                    if not chunk:
                        break
                    yield TTSAudioRawFrame(chunk, sample_rate, 1, context_id=context_id)

            async with self._client.audio.speech.with_streaming_response.create(**create_params) as r:
                if r.status_code != 200:
                    error = await r.text()
                    yield ErrorFrame(error=f"Error getting audio (status: {r.status_code}, error: {error})")
                    if proc.returncode is None:
                        proc.kill()
                    return

                await self.start_tts_usage_metrics(text)
                await self.stop_ttfb_metrics()

                async def write_stdin():
                    try:
                        async for chunk in r.iter_bytes(self.chunk_size):
                            if chunk and proc.stdin:
                                proc.stdin.write(chunk)
                                await proc.stdin.drain()
                    finally:
                        if proc.stdin:
                            proc.stdin.close()
                
                import asyncio
                writer_task = asyncio.create_task(write_stdin())

                async for frame in read_stdout():
                    yield frame
                
                await writer_task
                await proc.wait()

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            yield ErrorFrame(error=str(e))


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Run the voice bot for this session."""
    logger.info("Starting bot")

    # Mify LLM Config
    mify_api_key = os.getenv("MIFY_API_KEY", "dummy")
    mify_base_url = os.getenv("MIFY_BASE_URL", "http://model.mify.ai.srv/v1")

    # Speech-to-Text service (Self-hosted SenseVoice)
    # Using 'Language.ZH' to ensure SenseVoice knows we are expecting Chinese input
    stt = OpenAISTTService(
        api_key=os.getenv("STT_API_KEY", "dummy"),
        base_url=os.getenv("STT_BASE_URL", "http://red:5160/v1"),
        settings=OpenAISTTService.Settings(
            model=os.getenv("STT_MODEL", "sensevoice"),
            language=Language.ZH,
        )
    )

    # Text-to-Speech service (Self-hosted TTS) using the Custom class to decode the mp3 payload
    # Pipecat transports generally expect 16000Hz (webrtc default), OpenAI defaults to 24000
    tts = CustomOpenAITTSService(
        api_key=os.getenv("TTS_API_KEY", "dummy"),
        base_url=os.getenv("TTS_BASE_URL", "http://red:5050/v1"),
        sample_rate=16000,
        settings=OpenAITTSService.Settings(
            voice=os.getenv("TTS_MODEL", "zh-CN-XiaoxiaoNeural"),
            model=os.getenv("TTS_MODEL", "zh-CN-XiaoxiaoNeural"),
        )
    )

    # LLM service (Mify)
    llm = OpenAILLMService(
        api_key=mify_api_key,
        base_url=mify_base_url,
        settings=OpenAILLMService.Settings(
            model=os.getenv("MIFY_LLM_MODEL", "xiaomi/mimo-v2.5"),
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way. Answer in Chinese by default.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Pipeline - assembled from reusable components
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation
        context.add_message(
            {"role": "developer", "content": "Start by concisely introducing yourself in Chinese."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
