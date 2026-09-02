package dev.sakus.muscriptoreasy.model

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.zip.ZipInputStream

/** A validated, app-private MuScriptor Android model bundle. */
data class ModelBundle(
    val directory: File,
    val sourceName: String,
    val variant: String,
    val dim: Int,
    val card: Int,
    val maxContext: Int,
    val conditioner: File,
    val embedder: File,
    val decoder: File,
) {
    val totalBytes: Long
        get() = conditioner.length() + embedder.length() + decoder.length()
}

class ModelStore(private val context: Context) {
    private val modelDir: File = File(context.filesDir, "models").apply { mkdirs() }

    /**
     * Import a `.msa` zip into private storage.
     *
     * Only the four ABI-v1 root files are accepted. This intentionally rejects
     * arbitrary paths/directories, which also prevents zip-slip extraction.
     */
    fun importBundle(uri: Uri): ModelBundle {
        val displayName = queryDisplayName(uri) ?: "muscriptor-model.msa"
        require(displayName.lowercase().endsWith(".msa")) {
            "MuScriptor Android model must be a .msa bundle"
        }

        val safeStem = displayName
            .removeSuffix(".msa")
            .removeSuffix(".MSA")
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
            .ifBlank { "muscriptor-model" }

        val temp = File(modelDir, ".import-${UUID.randomUUID()}")
        require(temp.mkdirs()) { "Unable to create model import directory" }

        try {
            extractBundle(uri, temp)
            val validated = validateBundle(temp, displayName)

            val target = File(modelDir, safeStem)
            if (target.exists()) require(target.deleteRecursively()) {
                "Unable to replace existing model bundle"
            }
            require(temp.renameTo(target)) { "Unable to finalize imported model bundle" }
            return validateBundle(target, displayName)
        } catch (t: Throwable) {
            temp.deleteRecursively()
            throw t
        }
    }

    fun installedBundles(): List<ModelBundle> =
        modelDir.listFiles()
            ?.filter { it.isDirectory && !it.name.startsWith(".import-") }
            ?.mapNotNull { runCatching { validateBundle(it, it.name + ".msa") }.getOrNull() }
            ?.sortedBy { it.sourceName.lowercase() }
            .orEmpty()

    /** Kept temporarily for low-level ExecuTorch bring-up/debugging. */
    fun importPte(uri: Uri): File {
        val displayName = queryDisplayName(uri) ?: "muscriptor-model.pte"
        require(displayName.lowercase().endsWith(".pte")) {
            "ExecuTorch model must be a .pte file"
        }

        val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val output = File(modelDir, safeName)
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected model" }
            output.outputStream().use { out -> input.copyTo(out) }
        }
        return output
    }

    private fun extractBundle(uri: Uri, outputDir: File) {
        val allowed = setOf("manifest.json", "conditioner.pte", "embedder.pte", "decoder.pte")
        val seen = mutableSetOf<String>()
        var totalExtracted = 0L

        context.contentResolver.openInputStream(uri).use { raw ->
            requireNotNull(raw) { "Unable to open selected model bundle" }
            ZipInputStream(raw.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    require(!entry.isDirectory) { "Bundle directories are not allowed" }
                    require(entry.name in allowed) { "Unexpected bundle entry: ${entry.name}" }
                    require(seen.add(entry.name)) { "Duplicate bundle entry: ${entry.name}" }

                    val outFile = File(outputDir, entry.name)
                    outFile.outputStream().buffered().use { out ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val read = zip.read(buffer)
                            if (read < 0) break
                            totalExtracted += read
                            require(totalExtracted <= MAX_EXTRACTED_BYTES) {
                                "Model bundle is larger than the supported extraction limit"
                            }
                            out.write(buffer, 0, read)
                        }
                    }
                    zip.closeEntry()
                }
            }
        }

        require(seen == allowed) {
            "Incomplete model bundle; expected ${allowed.sorted().joinToString()}"
        }
    }

    private fun validateBundle(dir: File, sourceName: String): ModelBundle {
        require(dir.isDirectory) { "Model bundle directory does not exist" }
        val manifestFile = File(dir, "manifest.json")
        require(manifestFile.isFile) { "Bundle is missing manifest.json" }
        val manifest = JSONObject(manifestFile.readText())

        require(manifest.getString("format") == "muscriptor-android-bundle") {
            "Unsupported model bundle format"
        }
        require(manifest.getInt("abi_version") == 1) {
            "Unsupported model bundle ABI: ${manifest.optInt("abi_version", -1)}"
        }

        val files = manifest.getJSONObject("files")
        val conditionerName = files.getString("conditioner")
        val embedderName = files.getString("embedder")
        val decoderName = files.getString("decoder")
        require(conditionerName == "conditioner.pte") { "Unexpected conditioner filename" }
        require(embedderName == "embedder.pte") { "Unexpected embedder filename" }
        require(decoderName == "decoder.pte") { "Unexpected decoder filename" }

        val conditioner = File(dir, conditionerName)
        val embedder = File(dir, embedderName)
        val decoder = File(dir, decoderName)
        require(conditioner.isFile && conditioner.length() > 0) { "Missing conditioner.pte" }
        require(embedder.isFile && embedder.length() > 0) { "Missing embedder.pte" }
        require(decoder.isFile && decoder.length() > 0) { "Missing decoder.pte" }

        val model = manifest.getJSONObject("model")
        val runtime = manifest.getJSONObject("runtime")
        val variant = model.getString("variant")
        val dim = model.getInt("dim")
        val card = model.getInt("card")
        val maxContext = runtime.getInt("max_context")
        require(dim > 0 && card >= 1393 && maxContext >= 504) {
            "Invalid model dimensions in manifest"
        }

        return ModelBundle(
            directory = dir,
            sourceName = sourceName,
            variant = variant,
            dim = dim,
            card = card,
            maxContext = maxContext,
            conditioner = conditioner,
            embedder = embedder,
            decoder = decoder,
        )
    }

    private fun queryDisplayName(uri: Uri): String? {
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) return cursor.getString(index)
        }
        return null
    }

    companion object {
        // Defensive zip-bomb limit. Current Small/Medium bundles are far below this.
        private const val MAX_EXTRACTED_BYTES = 8L * 1024 * 1024 * 1024
    }
}
