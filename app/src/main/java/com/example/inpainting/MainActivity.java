package com.example.inpainting;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import ai.onnxruntime.OrtException;


public class MainActivity extends AppCompatActivity implements View.OnClickListener, InpaintingTask.TaskListener {

    private ActivityResultLauncher<Intent> resultLauncher;

    private ImageView imageView;

    private int imgWidth = 1024;

    private int imgHeight = 1024;
    private ImageButton confirmButton;

    private ImageButton cancelButton;

    private ImageButton openButton;

    private ImageButton saveButton;

    private ImageButton showButton;

    private  ImageButton backButton;

    private  ImageButton forwardButton;

    private SeekBar seekBar;

    private TextView textView;

    private TextView brush_text;

    private Bitmap mask = null;

    private Bitmap resMask;

    private Bitmap src_img;

    private ArrayList<Bitmap> bitmapArray;  // save the inpainting results

    private ArrayList<Bitmap> maskedArray;  // save the drawing process

    private ArrayList<Bitmap> maskArray;  // save the drawing mask

    private ArrayList<Bitmap> allArray;   // save all bitmaps

    private int current_idx = 0;

    private Canvas drawCanvas;

    private Canvas maskCanvas;

    private Paint paint;

    private float startX, startY, endX, endY;

    private int brushSize = 75;
    private boolean isDrawing = false;

    private  boolean debug = true;

    private InpaintingModel inpaintingModel;

    private static final String TAG = "MainActivity";



    @SuppressLint({"ClickableViewAccessibility", "MissingInflatedId"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            Resources resources = getResources();
            inpaintingModel = new InpaintingModel(resources);

        } catch (OrtException | IOException e) {
            throw new RuntimeException(e);
        }

        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        confirmButton = findViewById(R.id.button_confirm);
        cancelButton = findViewById(R.id.button_clear);
        saveButton = findViewById(R.id.button_save);
        openButton = findViewById(R.id.button_open);
        backButton = findViewById(R.id.button_back);
        forwardButton = findViewById(R.id.button_forth);
        showButton = findViewById(R.id.button_show);
        textView = findViewById(R.id.text);
        seekBar = findViewById(R.id.seekBar);
        brush_text = findViewById(R.id.seekBar_progress);
        seekBar.setProgress(brushSize);
        brush_text.setText(String.valueOf(brushSize));


        // 加载图片资源并设置给ImageView
        src_img = BitmapFactory.decodeResource(getResources(), R.drawable.input);
        init_res();

        paint = new Paint();
        paint.setAntiAlias(true);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            paint.setColor(Color.argb(0.2f,0.f,0.f,0.8f));
        }
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setStrokeWidth(brushSize);

        imageView.setOnTouchListener((v, event) -> {
            int action = event.getAction();
            switch (action) {
                case MotionEvent.ACTION_DOWN:
                    updateArray();
                    Bitmap copyBitmap_ = allArray.get(allArray.size() - 1).copy(Bitmap.Config.ARGB_8888,true);
                    resetCanvas(copyBitmap_);

                    // 记录起始点位置，并开始涂抹
                    startX = event.getX();
                    startY = event.getY();

                    drawCanvas.drawCircle(startX,startY,brushSize / 8.0f,paint);
                    maskCanvas.drawCircle(startX,startY,brushSize / 8.0f,paint);
                    isDrawing = true;
                    Log.d(TAG, "touch image");
                    Log.d(TAG, "isDrawing: " + isDrawing);
                    Log.d(TAG, "startX: " + startX);
                    Log.d(TAG, "startY: " + startY);

                    imageView.invalidate();
                    break;
                case MotionEvent.ACTION_MOVE:
                    if (isDrawing) {
                        Log.d(TAG, "drawing image");
                        // 记录当前点位置，并在drawCanvas上绘制路径
                        endX = event.getX();
                        endY = event.getY();
                        Log.d(TAG, "endX: " + endX);
                        Log.d(TAG, "endY: " + endY);
                        drawCanvas.drawLine(startX, startY, endX, endY, paint);
                        maskCanvas.drawLine(startX, startY, endX, endY, paint);
                        imageView.invalidate();

                        startX = endX;
                        startY = endY;
                        Log.d(TAG, "isDrawing: " + isDrawing);
                    }
                    break;
                case MotionEvent.ACTION_UP:
                    if (isDrawing) {
                        isDrawing = false;
                    }
                    imageView.invalidate();
                    Log.d(TAG, "isDrawing: " + isDrawing);
                    handle_button_color();

                    break;

                default:
                    break;
            }
            return true;
        });

        confirmButton.setOnClickListener(this);
        cancelButton.setOnClickListener(this);
        saveButton.setOnClickListener(this);
        openButton.setOnClickListener(this);
        backButton.setOnClickListener(this);
        forwardButton.setOnClickListener(this);
        showButton.setOnClickListener(this);


        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // 在此处处理 SeekBar 进度的更改
                brushSize = progress ;
                paint.setStrokeWidth(brushSize);
                brush_text.setText(String.valueOf(brushSize));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // 在此处处理 SeekBar 触摸开始的事件
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // 在此处处理 SeekBar 触摸停止的事件
            }
        });


        resultLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult result) {

                if (result.getResultCode() == RESULT_OK){
                    Intent intent = result.getData();
                    Uri picUri = intent.getData();
                    if (picUri != null){
                        InputStream inputStream;
                        try {
                            inputStream = getContentResolver().openInputStream(picUri);
                        } catch (FileNotFoundException e) {
                            throw new RuntimeException(e);
                        }
                        src_img = BitmapFactory.decodeStream(inputStream);
                        init_res();
                    }
                }

            }
        });


        showButton.setOnLongClickListener(v -> {
            showSrcimg();
            return true;
        });

        showButton.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                unshowSrcimg();
            }
            return false;
        });
    }

    private void init_res(){
        src_img = Bitmap.createScaledBitmap(src_img,imgWidth,imgHeight,true);

        bitmapArray = new ArrayList<>();
        bitmapArray.add(src_img);

        maskedArray = new ArrayList<>();
        maskArray = new ArrayList<>();

        allArray = new ArrayList<>();
        allArray.addAll(bitmapArray);
        current_idx = allArray.size() - 1 ;

        imageView.setImageBitmap(src_img.copy(Bitmap.Config.ARGB_8888,true));
    }


    private void resetCanvas(Bitmap bitmap){
        Bitmap maskBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        maskedArray.add(bitmap);
        maskArray.add(maskBitmap);
        drawCanvas = new Canvas(maskedArray.get(maskedArray.size() - 1));
        maskCanvas = new Canvas(maskArray.get(maskArray.size() - 1));
        imageView.setImageBitmap(bitmap);
        imageView.invalidate();
        allArray = new ArrayList<>();
        allArray.addAll(bitmapArray);
        allArray.addAll(maskedArray);
        current_idx = allArray.size() - 1 ;
    }

    private void createMaskBitmap() {
        Bitmap tmp = bitmapArray.get(bitmapArray.size() - 1);
        Bitmap blackbitmap = Bitmap.createBitmap(tmp.getWidth(), tmp.getHeight(), Bitmap.Config.ARGB_8888);
        Bitmap maskbitmap = Bitmap.createBitmap(tmp.getWidth(), tmp.getHeight(), Bitmap.Config.ARGB_8888);

        Canvas canvas = new Canvas(blackbitmap);
        canvas.drawColor(Color.BLACK);

        canvas.setBitmap(maskbitmap);

        for (int i = 0; i < maskArray.size() ; i++ ){
            canvas.drawBitmap(maskArray.get(i), 0, 0, null);
        }

        Paint paint = new Paint();
        paint.setAntiAlias(true);
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_IN));
        canvas.drawBitmap(blackbitmap, 0, 0, paint);

        mask = maskbitmap.copy(Bitmap.Config.ARGB_8888,true);

    }

    private void saveImg(String path, Bitmap bitmap){
        FileOutputStream fos = null;
        try{
            fos = new FileOutputStream(path);
            bitmap.compress(Bitmap.CompressFormat.JPEG,100,fos);
            MediaStore.Images.Media.insertImage(getContentResolver(), bitmap, "Inpainting_" + System.currentTimeMillis(), "Inpainting result.");
        } catch (Exception e){
            e.printStackTrace();
            new AlertDialog.Builder(this)
                    .setTitle("Save result")
                    .setMessage("Failed to save the image!")
                    .setNegativeButton("OK", null)
                    .show();
        } finally {
            if (fos != null){
                try{
                    fos.close();
                }catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
        Log.d(TAG,"The image has been saved to: " + path);
        new AlertDialog.Builder(this)
                .setTitle("Save result")
                .setMessage("Successfully saved the image!")
                .setPositiveButton("OK", null)
                .show();

    }

    private void doConfirm(){
        updateArray();
        if (! maskArray.isEmpty()){
            createMaskBitmap();
            InpaintingTask inpaintingTask = new InpaintingTask(this, textView);
            inpaintingTask.execute(inpaintingModel,bitmapArray.get(bitmapArray.size() - 1),mask); // 传递参数
        }

    }

    private void updateArray(){
        //when the img is being modified, current behavior should be the last element in array.
        if (current_idx  <= bitmapArray.size() - 1){
            bitmapArray = new ArrayList<>(allArray.subList(0, current_idx + 1)) ;
            maskedArray.clear();
            maskArray.clear();
        } else if (current_idx < allArray.size() - 1) {
            maskedArray = new ArrayList<>(allArray.subList(bitmapArray.size(), current_idx + 1)) ;
            maskArray = new ArrayList<>(maskArray.subList(0, maskedArray.size())) ;
        }

        allArray = new ArrayList<>();
        allArray.addAll(bitmapArray);
        allArray.addAll(maskedArray);
        current_idx = allArray.size() - 1 ;
    }

    private void doClear(){
        current_idx = bitmapArray.size() - 1;
        updateArray();
        handle_button_color();
        resetCanvas(bitmapArray.get(bitmapArray.size() - 1).copy(Bitmap.Config.ARGB_8888,true));
    }

    private void showSrcimg(){
        Bitmap show = BitmapFactory.decodeResource(getResources(), R.drawable.show);
        showButton.setImageBitmap(show);
        imageView.setImageBitmap(src_img);
    }

    private void unshowSrcimg(){
        Bitmap unshow = BitmapFactory.decodeResource(getResources(), R.drawable.unshow);
        showButton.setImageBitmap(unshow);
        imageView.setImageBitmap(allArray.get(current_idx));
    }

    private void handle_button_color(){
        if (current_idx > 0){
            backButton.setImageResource(R.drawable.back);

        }else{
            backButton.setImageResource(R.drawable.back_gray);
        }

        if (current_idx < allArray.size() - 1){
            forwardButton.setImageResource(R.drawable.forward);
        }else{
            forwardButton.setImageResource(R.drawable.forward_gray);
        }

        if (maskArray.isEmpty()){
            cancelButton.setImageResource(R.drawable.undo_gray);
        }else {
            cancelButton.setImageResource(R.drawable.undo);
        }
    }

    private void doBackward(){
        current_idx = Math.max(current_idx - 1, 0);
        imageView.setImageBitmap(allArray.get(current_idx));
        handle_button_color();
    }

    private void doForward(){
        current_idx = Math.min(current_idx + 1, allArray.size() - 1);
        imageView.setImageBitmap(allArray.get(current_idx));
        handle_button_color();
    }

    @Override
    public void onClick(View v) {
        
        if (v.getId() == R.id.button_confirm){
            confirmButton.setImageResource(R.drawable.process);
            doConfirm();
            confirmButton.setImageResource(R.drawable.ok);
        } else if (v.getId() == R.id.button_clear) {
            doClear();
        } else if (v.getId() == R.id.button_save) {
            String fileName = "Inpainting_" + System.currentTimeMillis() + ".jpeg";
            String path = getExternalFilesDir(Environment.DIRECTORY_PICTURES).toString() + File.separatorChar + fileName;
            saveImg(path,allArray.get(current_idx));
        } else if (v.getId() == R.id.button_open) {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            resultLauncher.launch(intent);
        } else if (v.getId() == R.id.button_back) {
            doBackward();
        } else if (v.getId() == R.id.button_forth) {
            doForward();
        } else{
            Log.d(TAG,"Unknown Button.");
        }

    }

    @Override
    public void onTaskCompleted(Bitmap[] result) {
        Bitmap src_img_temp = result[0].copy(Bitmap.Config.ARGB_8888,true);
        bitmapArray.add(src_img_temp);
        Bitmap copyBitmap = src_img_temp.copy(Bitmap.Config.ARGB_8888,true);
        maskedArray.clear();
        maskArray.clear();
        mask = null;
        resetCanvas(copyBitmap);
    }
}