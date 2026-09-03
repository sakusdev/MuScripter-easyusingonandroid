package dev.sakus.muscriptoreasy.inference

import dev.sakus.muscriptoreasy.model.ModelBundle
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.io.File

/** Android-side owner of the ExecuTorch modules used by MuScriptor ABI v1. */
class LocalMuScriptorEngine : Closeable {
    private var conditioner: Module? = null
    private var embedder: Module? = null
    private var decoder: Module? = null
    private var bundle: ModelBundle? = null

    // Temporary legacy single-module path retained for low-level runtime tests.
    private var legacyModule: Module? = null
    private var legacyModelFile: File? = null

    val isLoaded: Boolean
        get() = conditioner != null && embedder != null && decoder != null

    val loadedModelName: String?
        get() = bundle?.sourceName ?: legacyModelFile?.name

    val loadedVariant: String?
        get() = bundle?.variant

    /**
     * Load all ABI-v1 modules using mmap. New modules are fully opened before
     * the previous model is closed, so a failed import does not destroy a
     * working session.
     */
    fun load(modelBundle: ModelBundle): Result<Unit> = runCatching {
        val newConditioner = Module.load(
            modelBundle.conditioner.absolutePath,
            Module.LOAD_MODE_MMAP,
        )
        var newEmbedder: Module? = null
        var newDecoder: Module? = null
        try {
            newEmbedder = Module.load(
                modelBundle.embedder.absolutePath,
                Module.LOAD_MODE_MMAP,
            )
            newDecoder = Module.load(
                modelBundle.decoder.absolutePath,
                Module.LOAD_MODE_MMAP,
            )
        } catch (t: Throwable) {
            runCatching { newConditioner.close() }
            runCatching { newEmbedder?.close() }
            runCatching { newDecoder?.close() }
            throw t
        }

        closeModules()
        conditioner = newConditioner
        embedder = newEmbedder
        decoder = newDecoder
        bundle = modelBundle
        legacyModelFile = null
    }

    /** Temporary legacy `.pte` loader used only for runtime bring-up. */
    fun load(file: File): Result<Unit> = runCatching {
        require(file.isFile) { "Model does not exist: ${file.absolutePath}" }
        require(file.extension.equals("pte", ignoreCase = true)) { "Expected .pte model" }
        val loaded = Module.load(file.absolutePath, Module.LOAD_MODE_MMAP)
        closeModules()
        legacyModule = loaded
        legacyModelFile = file
        bundle = null
    }

    /** Run conditioner.pte: [1,501,512] log-mel -> [1,503,D] prefix. */
    fun conditionPrefix(
        logMel: FloatArray,
        instrumentToken: Long = 0,
        datasetToken: Long = 0,
    ): FloatArray {
        val model = requireBundle()
        require(logMel.size == MuScriptorLogMel.FRAMES * MuScriptorLogMel.MEL_BINS) {
            "Expected ${MuScriptorLogMel.FRAMES}x${MuScriptorLogMel.MEL_BINS} log-mel values"
        }

        val melTensor = Tensor.fromBlob(
            logMel,
            longArrayOf(1, MuScriptorLogMel.FRAMES.toLong(), MuScriptorLogMel.MEL_BINS.toLong()),
        )
        val instrumentTensor = Tensor.fromBlob(
            longArrayOf(instrumentToken),
            longArrayOf(1, 1),
        )
        val datasetTensor = Tensor.fromBlob(
            longArrayOf(datasetToken),
            longArrayOf(1, 1),
        )
        val output = requireConditioner().forward(
            EValue.from(melTensor),
            EValue.from(instrumentTensor),
            EValue.from(datasetTensor),
        ).single().toTensor()

        val expected = PREFIX_LENGTH * model.dim
        require(output.numel() == expected.toLong()) {
            "Conditioner returned ${output.numel()} values; expected $expected"
        }
        return output.dataAsFloatArray
    }

    /** Run embedder.pte for one upstream model token, including the initial card token. */
    fun embedToken(token: Int): FloatArray {
        val model = requireBundle()
        require(token in -1..model.card) { "Token $token is outside model embedding range" }
        val input = Tensor.fromBlob(longArrayOf(token.toLong()), longArrayOf(1, 1))
        val output = requireEmbedder().forward(EValue.from(input)).single().toTensor()
        require(output.numel() == model.dim.toLong()) {
            "Embedder returned ${output.numel()} values; expected ${model.dim}"
        }
        return output.dataAsFloatArray
    }

    /**
     * Run one or more contiguous stateful decoder positions through the same
     * `forward` method. Dynamic bundles accept up to 504 positions so the
     * condition prefix + initial token can be prefetched in a single call.
     * Legacy bundles still accept S=1 and are automatically kept on that path.
     */
    fun decodeEmbeddings(embeddings: FloatArray, startPos: Int): FloatArray {
        val model = requireBundle()
        require(embeddings.isNotEmpty() && embeddings.size % model.dim == 0) {
            "Decoder embedding buffer must contain complete ${model.dim}d rows"
        }
        val sequence = embeddings.size / model.dim
        require(sequence == 1 || model.supportsBlockPrefill) {
            "This model bundle only supports one decoder position at a time"
        }
        if (sequence > 1) {
            require(sequence <= model.maxPrefillSequence) {
                "Prefill length $sequence exceeds exported maximum ${model.maxPrefillSequence}"
            }
        }
        require(startPos >= 0 && startPos + sequence <= model.maxContext) {
            "Decoder positions $startPos..${startPos + sequence - 1} exceed context ${model.maxContext}"
        }

        val embeddingTensor = Tensor.fromBlob(
            embeddings,
            longArrayOf(1, sequence.toLong(), model.dim.toLong()),
        )
        val positions = LongArray(sequence) { i -> (startPos + i).toLong() }
        val positionTensor = Tensor.fromBlob(positions, longArrayOf(sequence.toLong()))
        val output = requireDecoder().forward(
            EValue.from(embeddingTensor),
            EValue.from(positionTensor),
        ).single().toTensor()
        require(output.numel() == model.card.toLong()) {
            "Decoder returned ${output.numel()} logits; expected ${model.card}"
        }
        return output.dataAsFloatArray
    }

    /** Run one stateful decoder position. */
    fun decodeEmbedding(embedding: FloatArray, inputPos: Int): FloatArray {
        val model = requireBundle()
        require(embedding.size == model.dim) {
            "Expected embedding dim ${model.dim}, got ${embedding.size}"
        }
        return decodeEmbeddings(embedding, inputPos)
    }

    /**
     * Prefill positions 0..503 with the 503 condition rows followed by the
     * initial model token. Returns logits for the initial token (the final row).
     */
    fun prefillConditionsAndInitial(prefix: FloatArray): FloatArray {
        val model = requireBundle()
        require(model.supportsBlockPrefill) { "Model does not support block prefill" }
        require(prefix.size == PREFIX_LENGTH * model.dim) {
            "Condition prefix has ${prefix.size} values; expected ${PREFIX_LENGTH * model.dim}"
        }
        val blockRows = PREFIX_LENGTH + 1
        require(blockRows <= model.maxPrefillSequence)

        val block = FloatArray(blockRows * model.dim)
        prefix.copyInto(block)
        val initial = embedToken(model.card)
        initial.copyInto(block, destinationOffset = prefix.size)
        return decodeEmbeddings(block, startPos = 0)
    }

    /** Greedy MT3 selection; reserved model logits above the generated vocab are ignored. */
    fun argmaxGenerated(logits: FloatArray): Int {
        val limit = minOf(logits.size, Mt3Vocab.GENERATED_VOCAB_SIZE)
        require(limit > 0) { "Empty logits" }
        var bestToken = 0
        var best = logits[0]
        for (token in 1 until limit) {
            if (logits[token] > best) {
                best = logits[token]
                bestToken = token
            }
        }
        return bestToken
    }

    fun requireConditioner(): Module =
        conditioner ?: error("No MuScriptor conditioner loaded")

    fun requireEmbedder(): Module =
        embedder ?: error("No MuScriptor token embedder loaded")

    fun requireDecoder(): Module =
        decoder ?: error("No MuScriptor decoder loaded")

    fun requireBundle(): ModelBundle =
        bundle ?: error("No MuScriptor Android bundle loaded")

    override fun close() {
        closeModules()
    }

    private fun closeModules() {
        runCatching { conditioner?.close() }
        runCatching { embedder?.close() }
        runCatching { decoder?.close() }
        runCatching { legacyModule?.close() }
        conditioner = null
        embedder = null
        decoder = null
        legacyModule = null
    }

    companion object {
        const val PREFIX_LENGTH = MuScriptorLogMel.FRAMES + 2
        const val BLOCK_PREFILL_LENGTH = PREFIX_LENGTH + 1
    }
}
