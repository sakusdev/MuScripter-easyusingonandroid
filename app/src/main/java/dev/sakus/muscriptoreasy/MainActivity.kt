package dev.sakus.muscriptoreasy

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.sakus.muscriptoreasy.inference.DecodedNoteEvent
import dev.sakus.muscriptoreasy.inference.LocalMuScriptorEngine
import dev.sakus.muscriptoreasy.inference.LocalTranscriptionResult
import dev.sakus.muscriptoreasy.inference.TranscriptionPipeline
import dev.sakus.muscriptoreasy.model.ModelStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MuScriptorHome()
                }
            }
        }
    }
}

@Composable
private fun MuScriptorHome() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val store = remember { ModelStore(context.applicationContext) }
    val engine = remember { LocalMuScriptorEngine() }
    val pipeline = remember { TranscriptionPipeline(context.applicationContext, engine) }

    DisposableEffect(engine) {
        onDispose { engine.close() }
    }

    var audioUri by remember { mutableStateOf<Uri?>(null) }
    var audioName by remember { mutableStateOf<String?>(null) }
    var modelStatus by remember { mutableStateOf("No local model bundle loaded") }
    var transcriptionStatus by remember { mutableStateOf("Ready when audio and model are selected") }
    var transcriptionResult by remember { mutableStateOf<LocalTranscriptionResult?>(null) }
    var isTranscribing by remember { mutableStateOf(false) }

    val audioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            audioUri = uri
            audioName = displayName(context, uri) ?: uri.lastPathSegment ?: "Selected audio"
            transcriptionResult = null
            transcriptionStatus = "Audio selected"
        }
    }

    val modelPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            modelStatus = "Importing model bundle…"
            scope.launch {
                val result = withContext(Dispatchers.IO) {
                    runCatching {
                        val bundle = store.importBundle(uri)
                        engine.load(bundle).getOrThrow()
                        bundle
                    }
                }
                modelStatus = result.fold(
                    onSuccess = {
                        "Loaded locally: ${it.variant} / ${it.dim}d " +
                            "(${formatBytes(it.totalBytes)}, context ${it.maxContext})"
                    },
                    onFailure = { "Model load failed: ${it.message}" },
                )
                transcriptionResult = null
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 28.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("MuScriptor Easy", style = MaterialTheme.typography.headlineMedium)
        Text(
            "100% local Android transcription. The app intentionally has no INTERNET permission.",
            style = MaterialTheme.typography.bodyMedium,
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Audio", style = MaterialTheme.typography.titleMedium)
                Text(audioName ?: "No audio selected")
                Button(
                    enabled = !isTranscribing,
                    onClick = { audioPicker.launch(arrayOf("audio/*", "video/*")) },
                ) {
                    Text("Choose audio / video")
                }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Local MuScriptor model", style = MaterialTheme.typography.titleMedium)
                Text(modelStatus)
                Button(
                    enabled = !isTranscribing,
                    onClick = {
                        modelPicker.launch(
                            arrayOf("application/zip", "application/octet-stream", "*/*"),
                        )
                    },
                ) {
                    Text("Import .msa bundle")
                }
            }
        }

        Spacer(Modifier.height(4.dp))

        Button(
            modifier = Modifier.fillMaxWidth(),
            enabled = audioUri != null && engine.isLoaded && !isTranscribing,
            onClick = {
                val selected = audioUri ?: return@Button
                isTranscribing = true
                transcriptionResult = null
                transcriptionStatus = "Starting…"
                scope.launch {
                    val result = withContext(Dispatchers.IO) {
                        runCatching {
                            pipeline.transcribe(selected) { progress ->
                                scope.launch {
                                    transcriptionStatus = if (progress.totalChunks > 0) {
                                        "${progress.stage}: ${progress.completedChunks}/${progress.totalChunks} chunks"
                                    } else {
                                        progress.stage
                                    }
                                }
                            }
                        }
                    }
                    isTranscribing = false
                    result.fold(
                        onSuccess = {
                            transcriptionResult = it
                            val starts = it.events.count { event -> event is DecodedNoteEvent.Start }
                            transcriptionStatus = buildString {
                                append("Done: ${it.chunks} chunks, ${it.generatedTokens} tokens, $starts note-ons")
                                if (it.chunksWithoutEos > 0) {
                                    append(", ${it.chunksWithoutEos} chunk(s) hit the token limit")
                                }
                            }
                        },
                        onFailure = {
                            transcriptionStatus = "Transcription failed: ${it.message}"
                        },
                    )
                }
            },
        ) {
            Text(if (isTranscribing) "Transcribing…" else "Transcribe")
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Result", style = MaterialTheme.typography.titleMedium)
                Text(transcriptionStatus)
                transcriptionResult?.let {
                    Text("Audio: %.2f s".format(it.audioDurationSeconds))
                    Text("Decoded events: ${it.events.size}")
                }
            }
        }

        Text(
            "Local PCM decode, upstream-style sinc resampling, log-mel, KV-cache generation and MT3 event decoding are wired. MIDI export is the next output step.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private fun displayName(context: Context, uri: Uri): String? {
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

private fun formatBytes(bytes: Long): String {
    val mib = bytes / (1024.0 * 1024.0)
    return if (mib >= 1024.0) "%.2f GiB".format(mib / 1024.0) else "%.1f MiB".format(mib)
}
