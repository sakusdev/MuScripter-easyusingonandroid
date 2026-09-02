package dev.sakus.muscriptoreasy.inference

import org.pytorch.executorch.Module
import java.io.File

/**
 * Android-side boundary for MuScriptor inference.
 *
 * v0.1 deliberately only owns model loading. The exported MuScriptor decoder
 * ABI (conditioning + token/position -> logits with model-owned KV caches)
 * will be wired here once the export tool is ready.
 */
class LocalMuScriptorEngine {
    private var module: Module? = null
    private var modelFile: File? = null

    val isLoaded: Boolean
        get() = module != null

    val loadedModelName: String?
        get() = modelFile?.name

    fun load(file: File): Result<Unit> = runCatching {
        require(file.isFile) { "Model does not exist: ${file.absolutePath}" }
        require(file.extension.equals("pte", ignoreCase = true)) { "Expected .pte model" }
        module = Module.load(file.absolutePath)
        modelFile = file
    }

    fun requireModule(): Module =
        module ?: error("No ExecuTorch model loaded")
}
