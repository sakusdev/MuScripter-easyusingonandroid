package dev.sakus.muscriptoreasy.inference

import dev.sakus.muscriptoreasy.model.ModelBundle
import org.pytorch.executorch.Module
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
}
