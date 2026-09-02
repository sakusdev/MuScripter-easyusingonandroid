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
     * Run one stateful decoder position. Calling position 0 starts a logically
     * fresh chunk because all future cache slots remain masked until overwritten.
     */
    fun decodeEmbedding(embedding: FloatArray, inputPos: Int): FloatArray {
        val model = requireBundle()
        require(embedding.size == model.dim) {
            "Expected embedding dim ${model.dim}, got ${embedding.size}"
        }
        require(inputPos in 0 until model.maxContext) {
            "Decoder position $inputPos exceeds context ${model.maxContext}"
        }

        val embeddingTensor = Tensor.fromBlob(
            embedding,
            longArrayOf(1, 1, model.dim.toLong()),
        )
        val positionTensor = Tensor.fromBlob(
            longArrayOf(inputPos.toLong()),
            longArrayOf(1),
        )
        val output = requireDecoder().forward(
            EValue.from(embeddingTensor),
            EValue.from(positionTensor),
        ).single().toTensor()
        require(output.numel() == model.card.toLong()) {
            "Decoder returned ${output.numel()} logits; expected ${model.card}"
        }
        return output.dataAsFloatArray
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
    }
}
