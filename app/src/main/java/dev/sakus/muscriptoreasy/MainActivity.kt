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
import dev.sakus.muscriptoreasy.inference.Mt3Instruments
import dev.sakus.muscriptoreasy.inference.TranscriptionPipeline
import dev.sakus.muscriptoreasy.midi.MidiExporter
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
    var pendingMidiBytes by remember { mutableStateOf<ByteArray?>(null) }
    var isTranscribing by remember { mutableStateOf(false) }
    var isPreparingMidi by remember { mutableStateOf(false) }

    val audioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            audioUri = uri
            audioName = displayName(context, uri) ?: uri.lastPathSegment ?: "Selected audio"
            transcriptionResult = null
            pendingMidiBytes = null
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
                pendingMidiBytes = null
            }
        }
    }

    val midiSaver = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("audio/midi"),
    ) { uri ->
        val bytes = pendingMidiBytes
        if (uri == null || bytes == null) {
            if (uri == null) transcriptionStatus = "MIDI save cancelled"
            return@rememberLauncherForActivityResult
        }
        scope.launch {
            val saved = withContext(Dispatchers.IO) {
                runCatching {
                    context.contentResolver.openOutputStream(uri, "w").use { output ->
                        requireNotNull(output) { "Unable to open MIDI destination" }
                        output.write(bytes)
                        output.flush()
                    }
                }
            }
            transcriptionStatus = saved.fold(
                onSuccess = { "MIDI saved (${formatBytes(bytes.size.toLong())})" },
                onFailure = { "MIDI save failed: ${it.message}" },
            )
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
                pendingMidiBytes = null
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
                transcriptionResult?.let { result ->
                    val instrumentNames = Mt3Instruments.programsIn(result.events)
                        .map(Mt3Instruments::nameForProgram)
                    Text("Audio: %.2f s".format(result.audioDurationSeconds))
                    Text("Decoded events: ${result.events.size}")
                    Text(
                        if (instrumentNames.isEmpty()) {
                            "Instruments: none detected"
                        } else {
                            "Instruments: ${instrumentNames.joinToString()}"
                        },
                    )
                    Button(
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !isTranscribing && !isPreparingMidi,
                        onClick = {
                            isPreparingMidi = true
                            transcriptionStatus = "Preparing MIDI…"
                            scope.launch {
                                val encoded = withContext(Dispatchers.Default) {
                                    runCatching { MidiExporter.encode(result.events) }
                                }
                                isPreparingMidi = false
                                encoded.fold(
                                    onSuccess = { bytes ->
                                        pendingMidiBytes = bytes
                                        transcriptionStatus = "MIDI ready (${formatBytes(bytes.size.toLong())})"
                                        midiSaver.launch(suggestMidiName(audioName))
                                    },
                                    onFailure = {
                                        transcriptionStatus = "MIDI export failed: ${it.message}"
                                    },
                                )
                            }
                        },
                    ) {
                        Text(if (isPreparingMidi) "Preparing MIDI…" else "Save MIDI")
                    }
                }
            }
        }

        Text(
            "Audio decode, upstream-style sinc resampling, log-mel, KV-cache generation, MT3 decoding and local type-1 MIDI export are wired end-to-end.",
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

private fun suggestMidiName(audioName: String?): String {
    val base = (audioName ?: "muscriptor-transcription")
        .substringBeforeLast('.', audioName ?: "muscriptor-transcription")
        .replace(Regex("[^A-Za-z0-9._ -]"), "_")
        .trim()
        .ifEmpty { "muscriptor-transcription" }
    return "$base.mid"
}

private fun formatBytes(bytes: Long): String {
    val kib = bytes / 1024.0
    val mib = kib / 1024.0
    return when {
        mib >= 1024.0 -> "%.2f GiB".format(mib / 1024.0)
        mib >= 1.0 -> "%.1f MiB".format(mib)
        kib >= 1.0 -> "%.1f KiB".format(kib)
        else -> "$bytes B"
    }
}
