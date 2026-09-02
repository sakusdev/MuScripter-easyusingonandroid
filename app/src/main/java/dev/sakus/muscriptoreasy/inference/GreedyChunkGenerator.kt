package dev.sakus.muscriptoreasy.inference

/** Result of one 5-second MuScriptor generation chunk. EOS itself is not included. */
data class ChunkGenerationResult(
    val tokens: List<Int>,
    val events: List<DecodedNoteEvent>,
    val reachedEos: Boolean,
    val forcedPreludeTokens: Int,
)

/**
 * Correctness-first ABI-v1 generation loop.
 *
 * The 503 conditioning embeddings are deliberately fed one at a time. This is
 * much slower than block prefill but gives us a static decoder shape and a very
 * simple desktop-vs-Android parity target.
 */
class GreedyChunkGenerator(
    private val engine: LocalMuScriptorEngine,
    private val mt3: Mt3StreamDecoder = Mt3StreamDecoder(),
) {
    /**
     * Generate one chunk from an already-computed [501,512] log-mel matrix.
     *
     * [forcePrelude] must be false for the first chunk and true for later chunks
     * when matching upstream's default `prelude_forcing=True` path.
     */
    fun generate(
        logMel: FloatArray,
        seekTimeSeconds: Double,
        nextSeekTimeSeconds: Double?,
        forcePrelude: Boolean,
        maxGenLen: Int = DEFAULT_MAX_GEN_LEN,
    ): ChunkGenerationResult {
        val model = engine.requireBundle()
        require(maxGenLen > 0) { "maxGenLen must be positive" }

        val events = mutableListOf<DecodedNoteEvent>()
        events += mt3.beginChunk(seekTimeSeconds, nextSeekTimeSeconds)

        // Important: beginChunk preserves normal open notes, so this is exactly
        // the decoder's view after processing the new boundary.
        val forcedPrompt = if (forcePrelude) mt3.tieSectionTokenIds() else IntArray(0)
        require(forcedPrompt.size < maxGenLen) {
            "Tie prelude (${forcedPrompt.size}) leaves no generation budget"
        }

        val prefix = engine.conditionPrefix(logMel)
        val dim = model.dim
        var position = 0
        val oneEmbedding = FloatArray(dim)
        var lastLogits: FloatArray? = null

        // Prefix positions 0..502. Their logits are intentionally ignored.
        for (prefixIndex in 0 until LocalMuScriptorEngine.PREFIX_LENGTH) {
            prefix.copyInto(
                oneEmbedding,
                startIndex = prefixIndex * dim,
                endIndex = (prefixIndex + 1) * dim,
            )
            lastLogits = engine.decodeEmbedding(oneEmbedding, position)
            position++
        }

        // Upstream generation sequence always starts with initial_token_id == card.
        lastLogits = engine.decodeEmbedding(engine.embedToken(model.card), position)
        position++

        // For chunks after the first, upstream passes tie_section_token_ids() as
        // `prompt`. Prompt tokens are yielded downstream and teacher-forced into
        // the transformer before the first free token is chosen.
        for (token in forcedPrompt) {
            events += mt3.feed(token)
            lastLogits = engine.decodeEmbedding(engine.embedToken(token), position)
            position++
        }

        val tokens = ArrayList<Int>()
        var reachedEos = false
        val freeBudget = maxGenLen - forcedPrompt.size

        for (step in 0 until freeBudget) {
            val token = engine.argmaxGenerated(checkNotNull(lastLogits))
            if (token == Mt3Vocab.EOS) {
                reachedEos = true
                break
            }

            tokens += token
            events += mt3.feed(token)

            // No reason to run one extra model step after the final allowed token.
            if (step + 1 < freeBudget) {
                require(position < model.maxContext) {
                    "Generation exceeded exported decoder context ${model.maxContext}"
                }
                lastLogits = engine.decodeEmbedding(engine.embedToken(token), position)
                position++
            }
        }

        return ChunkGenerationResult(
            tokens = tokens,
            events = events,
            reachedEos = reachedEos,
            forcedPreludeTokens = forcedPrompt.size,
        )
    }

    fun finish(): List<DecodedNoteEvent> = mt3.finish()

    companion object {
        const val DEFAULT_MAX_GEN_LEN = 2_000
    }
}
