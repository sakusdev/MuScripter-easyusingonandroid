package dev.sakus.muscriptoreasy.inference

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/** Decode an Android-supported audio/video URI to MuScriptor's 16 kHz mono float PCM. */
class AndroidAudioDecoder(private val context: Context) {
    data class AudioPcm(
        val samples: FloatArray,
        val sampleRate: Int,
    )

    fun decodeToMuScriptor(uri: Uri): FloatArray {
        val decoded = decodeMono(uri)
        return if (decoded.sampleRate == MuScriptorLogMel.SAMPLE_RATE) {
            decoded.samples
        } else {
            SincResampler(decoded.sampleRate, MuScriptorLogMel.SAMPLE_RATE)
                .process(decoded.samples)
        }
    }

    fun decodeMono(uri: Uri): AudioPcm {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = findAudioTrack(extractor)
            require(trackIndex >= 0) { "Selected file has no decodable audio track" }
            extractor.selectTrack(trackIndex)

            val inputFormat = extractor.getTrackFormat(trackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: error("Audio track has no MIME type")
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(inputFormat, null, null, 0)
            codec.start()

            var sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var pcmEncoding = if (inputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                inputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } else {
                AudioFormat.ENCODING_PCM_16BIT
            }

            val samples = FloatBuilder()
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val input = codec.getInputBuffer(inputIndex)
                            ?: error("Decoder returned null input buffer")
                        input.clear()
                        val size = extractor.readSampleData(input, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                size,
                                extractor.sampleTime.coerceAtLeast(0L),
                                extractor.sampleFlags,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val outputFormat = codec.outputFormat
                        if (outputFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                            sampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        }
                        if (outputFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                            channels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        }
                        pcmEncoding = if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                            outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        } else {
                            AudioFormat.ENCODING_PCM_16BIT
                        }
                    }
                    else -> if (outputIndex >= 0) {
                        if (info.size > 0) {
                            val output = codec.getOutputBuffer(outputIndex)
                                ?: error("Decoder returned null output buffer")
                            val bytes = output.duplicate().order(ByteOrder.nativeOrder())
                            bytes.position(info.offset)
                            bytes.limit(info.offset + info.size)
                            appendMono(samples, bytes.slice().order(ByteOrder.nativeOrder()), channels, pcmEncoding)
                        }
                        outputDone = (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        codec.releaseOutputBuffer(outputIndex, false)
                    }
                }
            }

            require(samples.size > 0) { "Decoder produced no PCM samples" }
            return AudioPcm(samples.toArray(), sampleRate)
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
            if (mime?.startsWith("audio/") == true) return i
        }
        return -1
    }

    private fun appendMono(
        destination: FloatBuilder,
        bytes: java.nio.ByteBuffer,
        channels: Int,
        pcmEncoding: Int,
    ) {
        require(channels > 0) { "Invalid channel count: $channels" }
        when (pcmEncoding) {
            AudioFormat.ENCODING_PCM_16BIT -> {
                val shorts = bytes.asShortBuffer()
                val frames = shorts.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) { sum += shorts.get() / 32768f }
                    destination.add(sum / channels)
                }
            }

            AudioFormat.ENCODING_PCM_FLOAT -> {
                val floats = bytes.asFloatBuffer()
                val frames = floats.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) { sum += floats.get() }
                    destination.add(sum / channels)
                }
            }

            AudioFormat.ENCODING_PCM_8BIT -> {
                val frames = bytes.remaining() / channels
                repeat(frames) {
                    var sum = 0f
                    repeat(channels) {
                        sum += ((bytes.get().toInt() and 0xff) - 128) / 128f
                    }
                    destination.add(sum / channels)
                }
            }

            else -> error("Unsupported decoder PCM encoding: $pcmEncoding")
        }
    }

    private class FloatBuilder(initialCapacity: Int = 64 * 1024) {
        private var data = FloatArray(initialCapacity)
        var size: Int = 0
            private set

        fun add(value: Float) {
            if (size == data.size) data = data.copyOf(data.size * 2)
            data[size++] = value
        }

        fun toArray(): FloatArray = data.copyOf(size)
    }

    companion object {
        private const val TIMEOUT_US = 10_000L
    }
}
