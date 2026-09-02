package dev.sakus.muscriptoreasy.inference

import android.content.Context
import android.net.Uri
import kotlin.math.ceil

/** Coarse progress suitable for the Compose UI. */
data class TranscriptionProgress(
    val stage: String,
    val completedChunks: Int = 0,
    val totalChunks: Int = 0,
)

data class LocalTranscriptionResult(
    val events: List<DecodedNoteEvent>,
    val generatedTokens: Int,
    val chunks: Int,
    val chunksWithoutEos: Int,
    val audioDurationSeconds: Double,
)

/** Full offline URI -> note-event pipeline. Must be called off the UI thread. */
class TranscriptionPipeline(
    context: Context,
    private val engine: LocalMuScriptorEngine,
) {
    private val audioDecoder = AndroidAudioDecoder(context.applicationContext)
    private val frontend = MuScriptorLogMel()

    fun transcribe(
        uri: Uri,
        onProgress: (TranscriptionProgress) -> Unit = {},
    ): LocalTranscriptionResult {
        check(engine.isLoaded) { "Load a MuScriptor Android model bundle first" }

        onProgress(TranscriptionProgress("Decoding audio"))
        val pcm = audioDecoder.decodeToMuScriptor(uri)
        val duration = pcm.size.toDouble() / MuScriptorLogMel.SAMPLE_RATE
        val totalChunks = ceil(pcm.size.toDouble() / MuScriptorLogMel.SEGMENT_SAMPLES)
            .toInt()
            .coerceAtLeast(1)

        val generator = GreedyChunkGenerator(engine)
        val allEvents = mutableListOf<DecodedNoteEvent>()
        var tokenCount = 0
        var noEos = 0

        for (chunkIndex in 0 until totalChunks) {
            onProgress(
                TranscriptionProgress(
                    stage = "Transcribing",
                    completedChunks = chunkIndex,
                    totalChunks = totalChunks,
                ),
            )

            val start = chunkIndex * MuScriptorLogMel.SEGMENT_SAMPLES
            val end = minOf(start + MuScriptorLogMel.SEGMENT_SAMPLES, pcm.size)
            val chunk = if (start < pcm.size) pcm.copyOfRange(start, end) else FloatArray(0)
            val logMel = frontend.compute(chunk)
            val seek = chunkIndex * 5.0
            val nextSeek = if (chunkIndex + 1 < totalChunks) (chunkIndex + 1) * 5.0 else null

            val result = generator.generate(
                logMel = logMel,
                seekTimeSeconds = seek,
                nextSeekTimeSeconds = nextSeek,
                forcePrelude = chunkIndex > 0,
            )
            allEvents += result.events
            tokenCount += result.tokens.size + result.forcedPreludeTokens
            if (!result.reachedEos) noEos++
        }

        allEvents += generator.finish()
        onProgress(
            TranscriptionProgress(
                stage = "Complete",
                completedChunks = totalChunks,
                totalChunks = totalChunks,
            ),
        )

        return LocalTranscriptionResult(
            events = allEvents,
            generatedTokens = tokenCount,
            chunks = totalChunks,
            chunksWithoutEos = noEos,
            audioDurationSeconds = duration,
        )
    }
}
