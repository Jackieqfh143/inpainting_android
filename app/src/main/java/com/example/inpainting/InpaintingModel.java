package com.example.inpainting;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Build;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;





public class InpaintingModel {
    private int imageHeight = 256;
    private int imageWidth = 256;


    private int ori_imageHeight;
    private int ori_imageWidth;


    private OrtEnvironment environment;
    private SessionOptions options;

    private Bitmap gt_img;
    private Bitmap gt_img_512;

    private Bitmap ori_gt_img;

    private Bitmap scaled_mask;
    private Bitmap scaled_mask_512;

    public long time_span;

    private int random_seed = 2023;

    private OrtSession mappingNetSession;
    private OrtSession encoderSession;
    private OrtSession generatorSession;
    private OrtSession srSession;
    private OrtSession miganSession;


    private Random rand_gen = new Random(random_seed);

    public InpaintingModel(Resources resources) throws OrtException, IOException {
        this.mappingNetSession = make_session(resources.openRawResource(R.raw.mapping));
        this.encoderSession = make_session(resources.openRawResource(R.raw.encoder));
        this.generatorSession = make_session(resources.openRawResource(R.raw.generator));
        this.srSession = make_session(resources.openRawResource(R.raw.esrgan));
        this.miganSession = make_session(resources.openRawResource(R.raw.migan));
    }


    private OrtSession make_session(InputStream inputStream) throws OrtException, IOException {
        //load the model
        byte[] modelBytes = new byte[inputStream.available()];
        inputStream.read(modelBytes);

        // 创建一个ONNX Runtime环境
        environment = OrtEnvironment.getEnvironment();

        // 配置ONNX Session的参数
        options = new SessionOptions();
        options.setIntraOpNumThreads(8);    //Sets the size of the CPU thread pool used for executing a single graph, if executing on a CPU.

        // 创建一个ONNX Session

        return environment.createSession(modelBytes, options);
    }

    public static Bitmap convertToGrayscale(Bitmap bitmap) {
        Bitmap grayscaleBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0);
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(filter);
        canvas.drawBitmap(bitmap, 0, 0, paint);
        return grayscaleBitmap;
    }

    public static int[] calculateHistogram(Bitmap grayscaleBitmap) {
        int[] histogram = new int[256]; // 256个灰度级别
        for (int i = 0; i < grayscaleBitmap.getWidth(); i++) {
            for (int j = 0; j < grayscaleBitmap.getHeight(); j++) {
                int pixel = grayscaleBitmap.getPixel(i, j);
                int grayValue = Color.red(pixel); // assuming the image is grayscale
                histogram[grayValue]++;
            }
        }
        return histogram;
    }

    public static int otsuThreshold(int[] histogram, int totalPixels) {
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += i * histogram[i];
        }

        float sumB = 0;
        int wB = 0;
        int wF = 0;
        float varMax = 0;
        int threshold = 0;

        for (int i = 0; i < 256; i++) {
            wB += histogram[i];
            if (wB == 0) {
                continue;
            }

            wF = totalPixels - wB;
            if (wF == 0) {
                break;
            }

            sumB += i * histogram[i];

            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;

            float varBetween = wB * wF * (mB - mF) * (mB - mF);

            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = i;
            }
        }

        return threshold;
    }

    public Bitmap getBinaryMask(Bitmap bitmap) {
        Bitmap graymap = convertToGrayscale(bitmap);
        int threshold = otsuThreshold(calculateHistogram(graymap), graymap.getWidth() * graymap.getHeight());
        //得到图形的宽度和长度
        int width = graymap.getWidth();
        int height = graymap.getHeight();
        //创建二值化图像
        Bitmap binarymap = null;
        binarymap = graymap.copy(Bitmap.Config.ARGB_8888, true);
        //依次循环，对图像的像素进行处理
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                //得到当前像素的值
                int col = binarymap.getPixel(i, j);
                //得到alpha通道的值
                int alpha = col & 0xFF000000;
                //得到图像的像素RGB的值
                int red = (col & 0x00FF0000) >> 16;
                int green = (col & 0x0000FF00) >> 8;
                int blue = (col & 0x000000FF);
                // 用公式X = 0.3×R+0.59×G+0.11×B计算出X代替原来的RGB
                int gray = (int) ((float) red * 0.3 + (float) green * 0.59 + (float) blue * 0.11);
                //对图像进行二值化处理
                if (gray <= threshold) {
                    gray = 0;
                } else {
                    gray = 255;
                }
                // 新的ARGB
                int newColor = alpha | (gray << 16) | (gray << 8) | gray;
                //设置新图像的当前像素值
                binarymap.setPixel(i, j, newColor);
            }
        }
        return binarymap;
    }

    private OnnxTensor bitmap2Tensor(Bitmap bitmap) throws OrtException{
        int channels = 3;
        int imageWidth = bitmap.getWidth();
        int imageHeight = bitmap.getHeight();
        FloatBuffer imgData = FloatBuffer.allocate(
                        channels
                        * imageWidth
                        * imageHeight
        );
        imgData.rewind();
        int stride = imageHeight * imageWidth;
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, imageWidth, 0, 0, imageWidth, imageHeight);

        // row first
        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < imageWidth; j++) {
                int idx = imageWidth * i + j;
                int pixelValue = bmpData[idx];
                imgData.put(idx, (pixelValue >> 16 & 0xFF) / 255.0f);   //R
                imgData.put(idx + stride, (pixelValue >> 8 & 0xFF) / 255.0f);   //G
                imgData.put(idx + stride * 2, (pixelValue & 0xFF) / 255.0f);   //B
            }
        }

        imgData.rewind();

        long[] target_shape = new long[]{1, channels, imageHeight, imageWidth};
        // 创建输入张量

        return OnnxTensor.createTensor(environment, imgData, target_shape);
    }

//    private OnnxTensor[] miganPreprocess(Bitmap img, Bitmap mask) throws OrtException{
//        Bitmap img_512 = Bitmap.createScaledBitmap(img, 512, 512, true);
//        Bitmap mask_512 = Bitmap.createScaledBitmap(mask, 512, 512, true);
//
//        int imageWidth = img_512.getWidth();
//        int imageHeight = img_512.getHeight();
//        IntBuffer imgBuffer = IntBuffer.allocate(
//                        3
//                        * imageWidth
//                        * imageHeight
//        );
//        imgBuffer.rewind();
//        int stride = imageHeight * imageWidth;
//        int[] imgData = new int[stride];
//        img_512.getPixels(imgData, 0, imageWidth, 0, 0, imageWidth, imageHeight);
//
//
//        IntBuffer maskBuffer = IntBuffer.allocate(
//                imageWidth * imageHeight
//        );
//        maskBuffer.rewind();
//        int[] maskData = new int[stride];
//        mask_512.getPixels(maskData, 0, imageWidth, 0, 0, imageWidth, imageHeight);
//
//        // row first
//        for (int i = 0; i < imageHeight; i++) {
//            for (int j = 0; j < imageWidth; j++) {
//                int idx = imageWidth * i + j;
//                int pixelValue =  imgData[idx];
//                int maskValue = ((maskData[idx] == Color.TRANSPARENT)  ? 255 : 0);
//                imgBuffer.put(idx, (pixelValue >> 16 & 0xFF));   //R
//                imgBuffer.put(idx + stride, (pixelValue >> 8 & 0xFF));   //G
//                imgBuffer.put(idx + stride * 2, (pixelValue & 0xFF));   //B
//
//                maskBuffer.put(idx, maskValue);
//            }
//        }
//
//        imgBuffer.rewind();
//        maskBuffer.rewind();
//
//        long[] img_shape = new long[]{1, 3, imageHeight, imageWidth};
//        long[] mask_shape = new long[]{1, 1, imageHeight, imageWidth};
//
//        // 创建输入张量
//
//        OnnxTensor imgTensor = OnnxTensor.createTensor(environment, imgBuffer, img_shape);
//        OnnxTensor maskTensor = OnnxTensor.createTensor(environment, maskBuffer, mask_shape);
//
//        return new OnnxTensor[]{imgTensor, maskTensor};
//    }

//    private Bitmap miganPostprocess(OnnxTensor onnxTensor){
//        // 获取张量的形状和数据
//        long[] shape = onnxTensor.getInfo().getShape();
//        IntBuffer buffer = onnxTensor.getIntBuffer();
//
//        int channels = (int) shape[1];
//        int height = (int) shape[2];
//        int width = (int) shape[3];
//        int[] pixels = new int[width * height];
//
//        // 假设数据是按照CHW格式存储的
//        for (int i = 0; i < height; i++) {
//            for (int j = 0; j < width; j++) {
//                int idx = width * i + j;
//                int r = buffer.get(idx);
//                int g = buffer.get(idx + width * height);
//                int b = buffer.get(idx + 2 * width * height);
//
//                // 将RGB值转换为像素值
//                pixels[idx] = (0xFF << 24) | (r << 16) | (g << 8) | b;
//            }
//        }
//
//        // 创建Bitmap
//        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
//        return bitmap;
//    }


    private Bitmap tensor2Bitmap(OnnxTensor tensor) {
        // 获取张量的形状和数据
        long[] shape = tensor.getInfo().getShape();
        FloatBuffer buffer = tensor.getFloatBuffer();

        int channels = (int) shape[1];
        int height = (int) shape[2];
        int width = (int) shape[3];
        int[] pixels = new int[width * height];

        // 假设数据是按照CHW格式存储的
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = width * i + j;
                int r = (int) (buffer.get(idx) * 255.0f);
                int g = (int) (buffer.get(idx + width * height) * 255.0f);
                int b = (int) (buffer.get(idx + 2 * width * height) * 255.0f);

                // 将RGB值转换为像素值
                pixels[idx] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            }
        }

        // 创建Bitmap
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }


    public OnnxTensor preprocess(Bitmap img, Bitmap mask, int channels) throws OrtException {
        this.ori_gt_img = img.copy(Bitmap.Config.ARGB_8888,true);
        this.ori_imageHeight = img.getHeight();
        this.ori_imageWidth = img.getWidth();
        this.gt_img = Bitmap.createScaledBitmap(img, imageWidth, imageHeight, true);
        this.scaled_mask = Bitmap.createScaledBitmap(mask, imageWidth, imageHeight, true);

        this.scaled_mask = getBinaryMask(this.scaled_mask);
        FloatBuffer imgData = FloatBuffer.allocate(
                        channels
                        * imageWidth
                        * imageHeight
        );
        imgData.rewind();
        int stride = imageHeight * imageWidth;
        int[] bmpData = new int[stride];
        int[] maskData = new int[stride];
        this.gt_img.getPixels(bmpData, 0, imageWidth, 0, 0, imageWidth, imageHeight);
        this.scaled_mask.getPixels(maskData, 0, imageWidth, 0, 0, imageWidth,imageHeight);

        // row first
        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < imageWidth; j++) {
                int idx = imageWidth * i + j;
                int pixelValue = bmpData[idx];
                float maskValue = (maskData[idx] == Color.TRANSPARENT)  ? 1.0f : 0.0f ;

                imgData.put(idx, (((pixelValue >> 16 & 0xFF) / 127.5f) - 1.0f) * maskValue);   //R
                imgData.put(idx + stride, (((pixelValue >> 8 & 0xFF) / 127.5f) - 1.0f) * maskValue); //G
                imgData.put(idx + stride * 2, (((pixelValue & 0xFF) / 127.5f) - 1.0f) * maskValue); //B
                imgData.put(idx + stride * 3, maskValue); //Mask
            }
        }

        imgData.rewind();

        long[] target_shape = new long[]{1, channels, imageHeight, imageWidth};
        // 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(environment, imgData, target_shape);

        return inputTensor;
    }

    public OnnxTensor miganPreprocess(Bitmap img, Bitmap mask, int channels) throws OrtException {
        this.ori_gt_img = img.copy(Bitmap.Config.ARGB_8888,true);

        this.ori_imageHeight = img.getHeight();
        this.ori_imageWidth = img.getWidth();
        gt_img_512 = Bitmap.createScaledBitmap(img, 512, 512, true);
        scaled_mask_512 = Bitmap.createScaledBitmap(mask, 512, 512, true);

        scaled_mask_512 = getBinaryMask(scaled_mask_512);
        FloatBuffer imgData = FloatBuffer.allocate(
                        channels
                        * 512
                        * 512
        );
        imgData.rewind();
        int stride = 512 * 512;
        int[] bmpData = new int[stride];
        int[] maskData = new int[stride];
        gt_img_512.getPixels(bmpData, 0, 512, 0, 0, 512, 512);
        scaled_mask_512.getPixels(maskData, 0, 512, 0, 0, 512,512);

        // row first
        for (int i = 0; i < 512; i++) {
            for (int j = 0; j < 512; j++) {
                int idx = 512 * i + j;
                int pixelValue = bmpData[idx];
                float maskValue = (maskData[idx] == Color.TRANSPARENT)  ? 1.0f : 0.0f ;
                imgData.put(idx, maskValue - 0.5f); //Mask

                imgData.put(idx + stride, (((pixelValue >> 16 & 0xFF) / 127.5f) - 1.0f) * maskValue);   //R
                imgData.put(idx + stride * 2, (((pixelValue >> 8 & 0xFF) / 127.5f) - 1.0f) * maskValue); //G
                imgData.put(idx + stride * 3, (((pixelValue & 0xFF) / 127.5f) - 1.0f) * maskValue); //B
            }
        }

        imgData.rewind();

        long[] target_shape = new long[]{1, channels, 512, 512};
        // 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(environment, imgData, target_shape);

        return inputTensor;
    }


    public Bitmap superResolution(Bitmap img) throws OrtException {
        OnnxTensor tensor = bitmap2Tensor(img);
        Map<String,OnnxTensor> in_dict = new HashMap<>();
        in_dict.put("input",tensor);
        OnnxTensor result = (OnnxTensor) srSession.run(in_dict).get(0);
        Bitmap resBitmap = tensor2Bitmap(result);
        return resBitmap;
    }

    public Bitmap[] postprocess(float[] out_array) throws OrtException {
        Bitmap comp_out_img = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        Bitmap in_mask = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        int stride = imageHeight * imageWidth;
        int[] imgData = new int[stride];
        int[] maskData = new int[stride];
        this.gt_img.getPixels(imgData, 0, imageWidth, 0, 0, imageWidth, imageHeight);
        this.scaled_mask.getPixels(maskData, 0, imageWidth, 0, 0, imageWidth,imageHeight);

        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < imageWidth; j++) {
                int idx = imageWidth * i + j;
                int gt_pixelValue = imgData[idx];
                float maskValue = (maskData[idx] == Color.TRANSPARENT)  ? 1.0f : 0.0f ; //0 for holes in mask
                float gt_R = (gt_pixelValue >> 16 & 0xFF) / 255f * maskValue;
                float gt_G = (gt_pixelValue >> 8 & 0xFF) / 255f  * maskValue;
                float gt_B = (gt_pixelValue & 0xFF) / 255f  * maskValue;

                float fake_R = (out_array[idx] + 1.0f) * 0.5f * (1.0f - maskValue);
                float fake_G = (out_array[idx + stride] + 1.0f) * 0.5f * (1.0f - maskValue);
                float fake_B = (out_array[idx + stride * 2] + 1.0f) * 0.5f * (1.0f - maskValue);

                float comp_R = (gt_R + fake_R);
                float comp_G = (gt_G + fake_G);
                float comp_B = (gt_B + fake_B);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    comp_out_img.setPixel(j,i,Color.argb(1.0f,comp_R,comp_G,comp_B));
                    int maskValue_ = (maskValue == 1.0f) ? Color.WHITE : Color.BLACK;
                    in_mask.setPixel(j,i,maskValue_);
                }

            }
        }

        // 使用超分辨率算法调整输出合成图像的分辨率
//        comp_out_img = superResolution(comp_out_img);
        comp_out_img = Bitmap.createScaledBitmap(comp_out_img, ori_imageWidth, ori_imageHeight, true);
        Bitmap out_mask  = Bitmap.createScaledBitmap(in_mask, ori_imageWidth, ori_imageHeight, true);
        stride = ori_imageHeight * ori_imageWidth;
        int[] imgData_ = new int[stride];
        int[] compImgData_ = new int[stride];
        int[] maskData_ = new int[stride];
        this.ori_gt_img.getPixels(imgData_, 0, ori_imageWidth, 0, 0, ori_imageWidth, ori_imageHeight);
        comp_out_img.getPixels(compImgData_, 0, ori_imageWidth, 0, 0, ori_imageWidth, ori_imageHeight);
        out_mask.getPixels(maskData_, 0, ori_imageWidth, 0, 0, ori_imageWidth,ori_imageHeight);

        for (int i = 0; i < ori_imageHeight; i++) {
            for (int j = 0; j < ori_imageWidth; j++) {
                int idx = ori_imageWidth * i + j;
                int gt_pixelValue = imgData_[idx];
                int fake_pixelValue = compImgData_[idx];
                float maskValue = (maskData_[idx] == Color.WHITE)  ? 1.0f : 0.0f ; //0 for holes in mask
                float gt_R = (gt_pixelValue >> 16 & 0xFF) / 255f * maskValue;
                float gt_G = (gt_pixelValue >> 8 & 0xFF) / 255f  * maskValue;
                float gt_B = (gt_pixelValue & 0xFF) / 255f  * maskValue;

                float fake_R = (fake_pixelValue >> 16 & 0xFF) / 255f * (1.0f - maskValue);
                float fake_G = (fake_pixelValue >> 8 & 0xFF) / 255f  * (1.0f - maskValue);
                float fake_B = (fake_pixelValue & 0xFF) / 255f  * (1.0f - maskValue);

                float comp_R = (gt_R + fake_R);
                float comp_G = (gt_G + fake_G);
                float comp_B = (gt_B + fake_B);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    comp_out_img.setPixel(j,i,Color.argb(1.0f,comp_R,comp_G,comp_B));
                }

            }
        }

        Bitmap[] out = new Bitmap[] {comp_out_img, in_mask};

        return out;
    }

    public Bitmap[] miganPostprocess(float[] out_array) throws OrtException {
        Bitmap comp_out_img = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888);
        Bitmap in_mask = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888);
        int stride = 512 * 512;
        int[] imgData = new int[stride];
        int[] maskData = new int[stride];
        gt_img_512.getPixels(imgData, 0, 512, 0, 0, 512, 512);
        scaled_mask_512.getPixels(maskData, 0, 512, 0, 0, 512,512);

        for (int i = 0; i < 512; i++) {
            for (int j = 0; j < 512; j++) {
                int idx = 512 * i + j;
                int gt_pixelValue = imgData[idx];
                float maskValue = (maskData[idx] == Color.TRANSPARENT)  ? 1.0f : 0.0f ; //0 for holes in mask
                float gt_R = (gt_pixelValue >> 16 & 0xFF) / 255f * maskValue;
                float gt_G = (gt_pixelValue >> 8 & 0xFF) / 255f  * maskValue;
                float gt_B = (gt_pixelValue & 0xFF) / 255f  * maskValue;

                float fake_R = (out_array[idx] + 1.0f) * 0.5f * (1.0f - maskValue);
                float fake_G = (out_array[idx + stride] + 1.0f) * 0.5f * (1.0f - maskValue);
                float fake_B = (out_array[idx + stride * 2] + 1.0f) * 0.5f * (1.0f - maskValue);

                float comp_R = (gt_R + fake_R);
                float comp_G = (gt_G + fake_G);
                float comp_B = (gt_B + fake_B);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    comp_out_img.setPixel(j,i,Color.argb(1.0f,comp_R,comp_G,comp_B));
                    int maskValue_ = (maskValue == 1.0f) ? Color.WHITE : Color.BLACK;
                    in_mask.setPixel(j,i,maskValue_);
                }

            }
        }

        // 使用超分辨率算法调整输出合成图像的分辨率
//        comp_out_img = superResolution(comp_out_img);
        comp_out_img = Bitmap.createScaledBitmap(comp_out_img, ori_imageWidth, ori_imageHeight, true);
        Bitmap out_mask  = Bitmap.createScaledBitmap(in_mask, ori_imageWidth, ori_imageHeight, true);
        stride = ori_imageHeight * ori_imageWidth;
        int[] imgData_ = new int[stride];
        int[] compImgData_ = new int[stride];
        int[] maskData_ = new int[stride];
        this.ori_gt_img.getPixels(imgData_, 0, ori_imageWidth, 0, 0, ori_imageWidth, ori_imageHeight);
        comp_out_img.getPixels(compImgData_, 0, ori_imageWidth, 0, 0, ori_imageWidth, ori_imageHeight);
        out_mask.getPixels(maskData_, 0, ori_imageWidth, 0, 0, ori_imageWidth,ori_imageHeight);

        for (int i = 0; i < ori_imageHeight; i++) {
            for (int j = 0; j < ori_imageWidth; j++) {
                int idx = ori_imageWidth * i + j;
                int gt_pixelValue = imgData_[idx];
                int fake_pixelValue = compImgData_[idx];
                float maskValue = (maskData_[idx] == Color.WHITE)  ? 1.0f : 0.0f ; //0 for holes in mask
                float gt_R = (gt_pixelValue >> 16 & 0xFF) / 255f * maskValue;
                float gt_G = (gt_pixelValue >> 8 & 0xFF) / 255f  * maskValue;
                float gt_B = (gt_pixelValue & 0xFF) / 255f  * maskValue;

                float fake_R = (fake_pixelValue >> 16 & 0xFF) / 255f * (1.0f - maskValue);
                float fake_G = (fake_pixelValue >> 8 & 0xFF) / 255f  * (1.0f - maskValue);
                float fake_B = (fake_pixelValue & 0xFF) / 255f  * (1.0f - maskValue);

                float comp_R = (gt_R + fake_R);
                float comp_G = (gt_G + fake_G);
                float comp_B = (gt_B + fake_B);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    comp_out_img.setPixel(j,i,Color.argb(1.0f,comp_R,comp_G,comp_B));
                }

            }
        }

        Bitmap[] out = new Bitmap[] {comp_out_img, in_mask};

        return out;
    }

    private OnnxTensor gen_rand_noise(int rows, int cols) throws OrtException{
        float[] arr = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[i * cols + j] = ((float) this.rand_gen.nextGaussian());
            }
        }

        FloatBuffer floatBuffer = FloatBuffer.wrap(arr);

        long[] target_shape = new long[]{rows,cols};

        floatBuffer.rewind();

        // 创建输入张量
        OnnxTensor noiseTensor = OnnxTensor.createTensor(environment, floatBuffer, target_shape);

        return noiseTensor;
    }


    public Bitmap[] miganInference(Bitmap inputImage, Bitmap mask) throws OrtException{
        OnnxTensor inputTensor = miganPreprocess(inputImage,mask,4);
        Map<String,OnnxTensor> in_dict = new HashMap<>();
        in_dict.put("input",inputTensor);

        OnnxTensor outputTensor = (OnnxTensor) miganSession.run(in_dict).get(0);
        return miganPostprocess(outputTensor.getFloatBuffer().array());
    }


    public Bitmap[] Inference(Bitmap inputImage, Bitmap mask) throws OrtException{
        OnnxTensor input = preprocess(inputImage,mask,4);

        OnnxTensor z = gen_rand_noise(1,512);
        long stime = System.nanoTime();
        OnnxTensor ws =  (OnnxTensor) mappingNetSession.run(Collections.singletonMap("noise", z)).get(0);
        Map<String,OnnxTensor> en_in_dict = new HashMap<>();
        en_in_dict.put("input",input);
        en_in_dict.put("in_ws",ws);
        Iterator<Map.Entry<String, OnnxValue>> en_out =  encoderSession.run(en_in_dict).iterator();

        Set<String> gen_input_names = generatorSession.getInputNames();

        Map<String,OnnxTensor> gen_in_dict = new HashMap<>();

        for (String name: gen_input_names){
            gen_in_dict.put(name,(OnnxTensor) en_out.next().getValue());
        }
        OnnxTensor outputTensor = (OnnxTensor) generatorSession.run(gen_in_dict).get(0);
        long etime = System.nanoTime();

        time_span = etime - stime;

        return postprocess(outputTensor.getFloatBuffer().array());

//        long stime = System.nanoTime();
//
//        Bitmap[] res = miganInference(inputImage, mask);
//        long etime = System.nanoTime();
//
//        time_span = etime - stime;
//
//        return res;

    }


}
