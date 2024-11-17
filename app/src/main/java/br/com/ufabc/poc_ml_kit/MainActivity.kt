package br.com.ufabc.poc_ml_kit

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.EditText
import android.widget.PopupMenu
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.canhub.cropper.CropImageView

class MainActivity : AppCompatActivity() {

    private lateinit var inputImageBtn: MaterialButton
    private lateinit var recognizeTextBtn: MaterialButton
    private lateinit var cropImageView: CropImageView
    private lateinit var recognizeTextEt: EditText
    private var imageUri: Uri? = null
    private lateinit var progressDialog: AlertDialog

    // ActivityResultLaunchers for gallery and camera
    private val pickImageFromGallery = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            imageUri = result.data?.data
            imageUri?.let { uri -> cropImageView.setImageUriAsync(uri) }
        }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success && imageUri != null) {
            cropImageView.setImageUriAsync(imageUri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageBtn = findViewById(R.id.inputImageBtn)
        recognizeTextBtn = findViewById(R.id.recognizeTextBtn)
        cropImageView = findViewById(R.id.cropImageView)
        recognizeTextEt = findViewById(R.id.recognizeTextEt)

        // Configure progress dialog
        progressDialog = AlertDialog.Builder(this).apply {
            setView(ProgressBar(this@MainActivity).apply { isIndeterminate = true })
            setCancelable(false)
        }.create()

        inputImageBtn.setOnClickListener { showInputImageDialog() }
        recognizeTextBtn.setOnClickListener { recognizeText() }

        cropImageView.setOnCropImageCompleteListener { _, result ->
            if (result.isSuccessful) {
                result.bitmap?.let {
                    cropImageView.setImageBitmap(it)
                    showToast("Image cropped successfully!")
                } ?: showToast("Cropped bitmap is null")
            } else {
                showToast("Crop failed: ${result.error}")
            }
        }
    }

    private fun showInputImageDialog() {
        PopupMenu(this, inputImageBtn).apply {
            menuInflater.inflate(R.menu.input_image_menu, menu)
            setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    R.id.menuCamera -> requestCameraPermission()
                    R.id.menuGallery -> openGallery()
                }
                true
            }
            show()
        }
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            showToast("Camera permission denied")
        }
    }

    private fun openCamera() {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "New Picture")
            put(MediaStore.Images.Media.DESCRIPTION, "From Camera")
        }
        imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        imageUri?.let { takePicture.launch(it) }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        pickImageFromGallery.launch(intent)
    }

    private fun recognizeText() {
        val bitmap = cropImageView.getCroppedImage()
        if (bitmap != null) {
            val inputImage = InputImage.fromBitmap(bitmap, 0)
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

            progressDialog.show()
            recognizer.process(inputImage)
                .addOnSuccessListener { visionText ->
                    progressDialog.dismiss()
                    recognizeTextEt.setText(visionText.text)
                }
                .addOnFailureListener { e ->
                    progressDialog.dismiss()
                    showToast("Failed to recognize text: ${e.message}")
                }
        } else {
            showToast("Please select an image first")
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
