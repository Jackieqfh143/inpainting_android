package com.example.inpainting;

import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Handler;
import android.os.Looper;
import android.widget.TextView;

import ai.onnxruntime.OrtException;

public class InpaintingTask extends AsyncTask<Object, Long, Bitmap[]> {

    private TextView textView;

    private long startTime;

    private Handler handler;

    private boolean isRunning = true;

    // 定义一个回调接口
    public interface TaskListener {
        void onTaskCompleted(Bitmap[] result);
    }

    private TaskListener taskListener;

    public InpaintingTask(TaskListener listener, TextView text) {
        this.taskListener = listener;
        this.textView = text;
        this.handler = new Handler(Looper.getMainLooper());
    }

    @Override
    protected Bitmap[] doInBackground(Object... objects) {
        startTime = System.nanoTime();
        InpaintingModel inpaintingModel = (InpaintingModel) objects[0];
        Bitmap img = (Bitmap) objects[1];
        Bitmap mask = (Bitmap) objects[2];
        Bitmap[] out;
        try {
            updateUIThread.start();
            out = inpaintingModel.Inference(img,mask);
        } catch (OrtException e) {
            e.printStackTrace();
            return new Bitmap[0];
        }
        return out; // 返回结果
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        // 在任务执行前，初始化界面状态
        textView.setText("Time cost: 0 ms");
    }

    @Override
    protected void onProgressUpdate(Long... values) {
        super.onProgressUpdate(values);
        long curTime = System.nanoTime() - startTime;
        String tmpString = "Time cost: " + curTime + " ms";
        textView.setText(tmpString);
    }

    @Override
    protected void onPostExecute(Bitmap[] result) {
        super.onPostExecute(result);
        // 在任务完成后，通过回调通知主线程
        if (taskListener != null) {
            isRunning = false;
            taskListener.onTaskCompleted(result);
        }
    }

    // 定义一个用于更新UI界面的线程
    private Thread updateUIThread = new Thread(new Runnable() {
        @Override
        public void run() {
            while (isRunning) {
                long curTime = (System.nanoTime() - startTime) / 1000000;
                final String tmpString = "Time cost: " + curTime + " ms";
                // 使用Handler将更新UI界面的任务post到主线程队列中执行
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        textView.setText(tmpString);
                    }
                });
                try {
                    Thread.sleep(50); // 每秒更新一次
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    });
}

