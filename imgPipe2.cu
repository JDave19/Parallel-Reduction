#include<opencv2/opencv.hpp>
#include <npp.h>
#include "wb.cuh"
#include<iostream>
#include "gtm.cuh"

// unsigned int *d_uintImg;

short *d_rawImg;
unsigned char  *d_init, *d_init3, *pDeviceBuffer, *d_init8;
unsigned short *h_cfa, *d_cfa, *d_init16;
short *d_l, *d_lSharp;
short *d_temp;
uchar *dummy;
unsigned int *d_out,*h_out;
// unsigned char *h_out;
unsigned int * d_pv_high, * d_pv_peak, * d_pvblock, * d_pv_pkadd, * h_pvhigh, * h_pvpeak, *d_pv_local;
float *wb;
int hpBufferSize;

////////// HDR vars //////////
unsigned char * msb_img_8UC1, * csb_img_8UC1, * lsb_img_8UC1;
unsigned char * d_merged_bayer_8UC1;

/////mblu and vector/////
float *dBb, hBb[9] =  {0, 0, 0, 0, 1, 0, 0, 0, 1};
uint *d_rgb, *tmpImg, *curImg;
cv::Mat cbcrchart, chartPic, chart_color;   // vector scope
unsigned int *d_chart, *d_opChart, *d_postRoi;  //vector scope
unsigned char *vYCbCr, *planarYCbCr[3];//vector scope
int hd_height = 1440, hd_width = 1440, hd_channels = 3;
uchar *d_sharpTemp, *hd_ycc8, *h_ycc8[3], **d_ycc8;
float *cbOff, *crOff, *d_hdr_weights, *d_white_matrix;
cv::Ptr<cv::CLAHE> clahe;
cv::Mat clIn, clOut;
float *d_plane1, *d_plane2;
unsigned char *d_l1;
Npp8u *d_ycbcrImage;
///////////////////////////

cudaArray *d_volumeArray = 0;
const cudaExtent volumeSize = make_cudaExtent(hd_width, hd_height, 1);
cudaTextureObject_t tex; 
cudaResourceDesc texRes;
cudaTextureDesc texDescr;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
cudaMemcpy3DParms copyParams = {0};

#define imgHei 1440
#define imgWid 1440

struct Params
{
    int imageW, imageH, sensorH, sensorW;
    float contrast, gam, chroma, rGain, bGain, gGain, wbGain, bright=50, zoom, cbrgam , bright_post, contrast_post, saturation_post, urGain, ubGain, ugGain;
    unsigned char wbFlag, frzCnt, maskType;
    int zoomx, zoomy, det, cor, deadpixMode, aecalc;
    bool preFrz, grtFlag, denSel, clahe, vec_scope, texMode, swap;
    float sharp_value, thresh, knn, klerpc, clip1, clip2, ffc_gain, hue_angle;
    int clrAdj[24], ffc_sigma, mblu, freeze_count, rSelect, gSelect, bSelect;
    int left, right, up, down, triangle;
    float slope, tresh;
    int judge;
    float temp1,temp2,temp3;

} *d_param, *h_par;

float plane1[72] = {1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    1,0,0,0,1,0,0,0,1,
                    0.509,0,0.784,0,1,0.572,0.211,1,0};


float plane2[36] = { 0,1,0,0,0,0.4,0,0,0.6, //0,1,1
                     0,1,0,0,0,0.4,0,0,0.55,//1,1,1
                     0,1,0,0,0,0.4,0,0,0.5,//0.8,0.8,0.8
                     0,1,0,0,0,0.4,0,0,0.4,//1,1,1
};

float *d_zoomVal, zoomVal[3] = {1, 1.2, 1.5}, *dWb, hWb[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, radius, sigma;

dim3 rSzgrid, blockSize(32, 32);
dim3 gridSz1D(2025, 1, 1), blkSz1D(1024, 1, 1), maskgridSz;
int imgH;

dim3 blockSize2D(32, 32);
dim3 gridSize2D((imgWid + blockSize2D.x - 1) / blockSize2D.x, (imgHei + blockSize2D.y - 1) / blockSize2D.y);

__global__ void cOffInit(float *cbOff, float *crOff)
{
    *cbOff = 0;
    *crOff = 0;
}

extern "C" void HDRWeights(float* h_hdr_weights)
{
    cudaMalloc((void**)&d_hdr_weights, 3 * 256 * sizeof(float));
    cudaMemcpy(d_hdr_weights, h_hdr_weights, 3 * 256 * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "HDR weights copied to device successfully!" << std::endl;
}
extern "C" void white_calib(float* h_white_matrix)
{
    cudaMalloc((void**)&d_white_matrix, imgHei * imgWid * sizeof(float));
    cudaMemcpy(d_white_matrix, h_white_matrix, imgHei * imgWid * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "White calibration copied to device successfully!" << std::endl;
}

extern "C" void initCuda_2mp(unsigned char *vec_chart)
{
    radius = 2; sigma = 0.5;

    cudaMalloc((void **)&d_param, sizeof(Params));
	cudaMalloc((void **)&d_rawImg, 744 * 744 * 5);
    cudaMalloc((void **)&d_cfa, hd_width * hd_height * sizeof(ushort));

    cudaMalloc((void **)&d_init16, hd_width * hd_height * hd_channels * sizeof(ushort));
    cudaMalloc((void **)&d_init8, hd_width * hd_height * hd_channels * sizeof(uchar));

    cudaMalloc((void **)&d_init, hd_width * hd_height * sizeof(int));
    cudaMalloc((void **)&d_init3, hd_width * hd_height * hd_channels * sizeof(uchar));
    //////////// HDR vars memory allocation //////////////
    cudaMalloc((void**)&msb_img_8UC1, hd_width * hd_height * sizeof(unsigned char));
    cudaMalloc((void**)&csb_img_8UC1, hd_width * hd_height * sizeof(unsigned char));
    cudaMalloc((void**)&lsb_img_8UC1, hd_width * hd_height * sizeof(unsigned char));

    cudaMalloc((void**)&d_merged_bayer_8UC1, hd_width * hd_height * sizeof(unsigned char));
    /////////////////////////////////////////////////////
    cudaMalloc((void **)&d_out, hd_width * hd_height * sizeof(int));   // cuda error change from above line
    cudaMalloc((void **)&d_zoomVal, 3 * sizeof(float));
    cudaMemcpy(d_zoomVal, zoomVal, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dWb, 9 * sizeof(float));
    cudaMemcpy(dWb, hWb, 9 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_l1, sizeof(unsigned char) * hd_width * hd_height);

    cudaMalloc((void **)&d_l, hd_width * hd_height * sizeof(short));
    cudaMalloc((void **)&d_lSharp, hd_width * hd_height * sizeof(short));
    cudaMalloc((void **)&d_temp, 1014 * sizeof(short));
    nppiFilterUnsharpGetBufferSize_16u_C1R (radius, sigma, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);

    cudaMalloc((void **)&d_pvblock, 2025 * sizeof(int));
    cudaMalloc((void **)&d_pv_pkadd, 2025 * sizeof(int));
    cudaMalloc((void **)&d_pv_high, sizeof(int));
    cudaMalloc((void **)&d_pv_peak, sizeof(int));

    h_pvpeak = (unsigned int *)malloc(sizeof(unsigned int));
    h_pvhigh = (unsigned int *)malloc(sizeof(unsigned int));
    h_cfa = (unsigned short *)malloc(sizeof(unsigned int));
    // cudaMalloc((void**)&d_uintImg, width * height * sizeof(unsigned int));

    // h_out = (unsigned int *)malloc(1080 * 1080 * 4);
    /*h_out = (unsigned int *)malloc(hd_width * hd_height * sizeof(int)); */  // cuda error change from above line
    h_out = (unsigned int *)malloc(hd_width * hd_height  * sizeof(int));
    dummy = (unsigned char *)malloc(hd_width * hd_height * 3 * sizeof(uchar));
    // memset(h_out, 128, 1080 * 1080 * 4);
    memset(h_out, 128, hd_width * hd_height * sizeof(int));        // cuda error change from above line

    wbInit(hd_width * hd_height );
	
    cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = d_volumeArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true; 
    texDescr.filterMode       = cudaFilterModeLinear; 
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    //////////////vector and mblu///////////
    /// \brief cudaMalloc
    ///

    cudaMalloc((void**)&hd_ycc8, sizeof(uchar) * hd_width * hd_height * 3);
    h_ycc8[0] = &hd_ycc8[0];
    h_ycc8[1] = &hd_ycc8[hd_width * hd_height];
    h_ycc8[2] = &hd_ycc8[hd_width * hd_height * 2];
    cudaMalloc((void**)&d_ycc8, sizeof(uchar *) * 3);
    cudaMemcpy(d_ycc8, h_ycc8, sizeof(uchar *) * 3, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cbOff, sizeof(float));
    cudaMalloc((void**)&crOff, sizeof(float));
    cOffInit<<<1,1>>>(cbOff, crOff);

    cudaMalloc((void**)&tmpImg, sizeof(uint) * hd_width * hd_height);//* 1440 * 1080);//
    cudaMalloc((void**)&curImg, sizeof(uint) * hd_width * hd_height);//* 1440 * 1080);//
    cudaMalloc((void**)&d_rgb, sizeof(uint) * hd_width * hd_height);

    cudaMalloc((void**)&dBb, 9 * sizeof(float));
    cudaMemcpy(dBb, hBb, sizeof(float) * 9, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_plane1, 99 * sizeof(float));
    cudaMalloc((void**)&d_plane2, 99 * sizeof(float));
    cudaMemcpy(d_plane1, plane1, 99*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_plane2, plane2, 99*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_l1, sizeof(unsigned char) * hd_width * hd_height);

    clahe = cv::createCLAHE();
    clahe->setTilesGridSize(cv::Size(4, 4));
    // clIn = cv::Mat(720, 960, CV_8UC1);
    // clOut = cv::Mat(720, 960, CV_8UC1);
    clIn = cv::Mat(hd_width, hd_height ,CV_16UC1);
    clOut = cv::Mat(hd_width, hd_height ,CV_16UC1);

    chartPic = cv::Mat(225, 225, CV_8UC4, vec_chart);
    cv::cvtColor(chartPic, cbcrchart, cv::COLOR_RGBA2BGRA);
    chart_color = cv::Mat(48,64,CV_8UC4);
    cudaMalloc((void**)&d_chart, 225 * 225 * sizeof(unsigned int));
    cudaMemcpy(d_chart, cbcrchart.data, 225 * 225 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_opChart, 225 * 225 * sizeof(unsigned int));
    cudaMalloc((void**)&d_postRoi, 64 * 48 * sizeof(unsigned int));
    cudaMalloc((void**)&vYCbCr, 64 * 48 * 3);
    for(int p = 0; p < 3; p++)
        planarYCbCr[p] = &vYCbCr[64 * 48 * p];
    cudaMalloc((void**)&d_ycbcrImage,hd_width *hd_height * hd_channels * sizeof(unsigned char));
}


__global__ void extractCFA8(unsigned char *in, unsigned short *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int loc = 1860 * (24 + (i/1440)) + 30 + ((i % 1440) / 4) * 5 + ((i % 1440) % 4);
    unsigned short bit = (in[loc + 4 - (i % 4)] >> (2 * (i % 4)) & 0x03);
    out[i] = (in[loc] << 2) + bit;
}

__inline__ void raw2cfa()
{
    extractCFA8 <<< 2025, 1024 >>> (reinterpret_cast<uchar *>(d_rawImg),d_cfa);
}
__global__ void blc (ushort *in, ushort *out, int blc_level, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        float scale = 1023.0f / (1023.0f - blc_level);  // Use float division
        int corrected_value = max(0, in[idx] - blc_level);  // Avoid negative values
        out[idx] = static_cast<ushort>(corrected_value * scale); // Clamp to 10-bit range
    }
}
__global__ void lsctilebased(ushort* in, ushort* out, float* white_img, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height)
    {
        float temp = static_cast<float>(in[idx]);
        float nume = temp - 10.0f;

        float denom = white_img[idx] - 10.0f;
        if (denom < 1e-5f) denom = 1.0f;

        float result = nume / denom ;

        result = result * 1023.f;
        result = fminf(1023.0f, fmaxf(0.0f, result));
        out[idx] = static_cast<ushort>(result);
    }
}
__global__ void lscdistancebased(ushort* in, ushort* out, float* white_img, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height)
    {
        int x = idx % width;  // Column index
        int y = idx / width;  // Row index

        int center_pixel_pos_x = (width >> 1);
        int center_pixel_pos_y = (height >> 1);

        int x_dist = x - center_pixel_pos_x;
        int y_dist = y - center_pixel_pos_y;

        float max_distance = sqrtf(center_pixel_pos_x * center_pixel_pos_x + center_pixel_pos_y * center_pixel_pos_y);
        float xy_distance = sqrtf(x_dist * x_dist + y_dist * y_dist);
        float distance = xy_distance / max_distance;

        // Gain formula with correct values
        float a = 0.01759f;
        float b = 28.37f;
        float c = 13.36f;

        // // Gain formula with correct values
        // float a = 0.0049f;
        // float b = 100.0f;
        // float c = 47.3f;

        float gain_val = (a * ((distance + b) * (distance + b))) - c;
        gain_val = fmaxf(gain_val, 0.0f);  // Ensure gain is non-negative

        out[idx] = static_cast<ushort>(in[idx] * gain_val);
    }
}

__global__ void gtm_test(ushort3 *in, uchar4 *out)
{
    // printf("First call from gtmtest");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i].x = (in[i].x) >> 2;
    out[i].y = (in[i].y) >> 2;
    out[i].z = (in[i].z) >> 2;
}

__global__ void bits_extractor(ushort* img, uchar* lsb_img, uchar* msb_img, uchar* csb_img, int _wid, int _hei)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * _wid + x;

    if (x < _wid && y < _hei) {
        // Convert 10-bit value to 8-bit and clamp it to the range [0, 255]
        lsb_img[idx] = (img[idx]    ) > 255 ? 255 : img[idx];
        csb_img[idx] = (img[idx] / 2) > 255 ? 255 : img[idx] / 2;
        msb_img[idx] = (img[idx] / 4) > 255 ? 255 : img[idx] / 4;

    }
}

__global__ void HDR(uchar* lsb_img, uchar* csb_img, uchar* msb_img, float* wr_ocv, uchar* ldr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * imgWid + x;

    // if(x< imgHei && y < imgWid)
    // {
    //     uchar val1 = lsb_img[idx];
    //     uchar val2 = csb_img[idx];
    //     uchar val3 = msb_img[idx];

    //     float sum = val1 + val2 + val3;
    //     // ldr[idx] = static_cast<uchar>(sum * 0.333f);
    //     ldr[idx] = csb_img[idx];
    // }

    if (x < imgHei && y < imgWid) {
        // Read pixel values
        uchar val1 = lsb_img[idx];
        uchar val2 = csb_img[idx];
        uchar val3 = msb_img[idx];

        // Compute weight factors
        float final_w1 = wr_ocv[val1] * 6.103515625e-05;
        float final_w2 = wr_ocv[256 + val2] * 6.103515625e-05;
        float final_w3 = wr_ocv[512 + val3] * 6.103515625e-05;

        // Weighted sum calculations
        float val_1 = final_w1 * val1;
        float val_2 = final_w2 * val2;
        float val_3 = final_w3 * val3;

        // Normalize by weight sum
        float sum_wei = final_w1 + final_w2 + final_w3;
        if (sum_wei == 0.0f) sum_wei = 1.0f; // Avoid division by zero

        int final_val = static_cast<int>((val_1 + val_2 + val_3) / sum_wei);

        // Clamping to 8-bit range
        final_val = max(0, min(255, final_val));

        // Store result
        ldr[idx] = static_cast<uchar>(final_val);
    }
}


__inline__ void gtm()
{
    // printf("First call from gtm");
    // gtm_test<<<2025, 1024>>>((ushort3 *)d_init16, (uchar4 *)d_init);
    // tone_mapping((uchar3 *)d_init8, (uchar4 *)d_init); //10 bit image is passed.
    // cudaDeviceSynchronize();
}

__global__ void c4Toc3(uchar4 *in, uchar3 *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i].x = in[i].z;
    out[i].y = in[i].y;
    out[i].z = in[i].x;
}

__global__ void c3Toc4(uchar3 *in, uchar4 *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i].x = in[i].z;
    out[i].y = in[i].y;
    out[i].z = in[i].x;
}
__global__ void c3Toc4copy(uchar3 *in, uchar4 *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i].x = in[i].x;
    out[i].y = in[i].y;
    out[i].z = in[i].z;
}

__global__ void extractL(uchar3 *lab, short *l)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    l[i] = (short)lab[i].x;
}

__global__ void FilterEdge(short *in, short *minV)//, unsigned short *maxV)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    in[i] = (short)clamp((int)in[i] - (int)minV[i], -16, 24);//-8, 12);
}

__global__ void EdgeAdd(short *in, short *minV)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // minV[i] = (short)clamp((int)in[i] + (int)minV[i], 0, 255);
    minV[i] = (short)clamp((int)in[i], 0, 255);
    //minV[i] = in[i];
}

__inline__ void sharpenYplane(float sharp_value)
{
    extractL<<<gridSz1D, blkSz1D>>>((uchar3 *)d_init3, d_l);
    nppiFilterUnsharpBorder_16s_C1R((short *)d_l, imgWid * 2, {0, 0}, (short *)d_lSharp, imgWid * 2, {imgWid, imgHei}, radius, sigma, sharp_value, 0.0f, NPP_BORDER_REPLICATE, pDeviceBuffer);

    // FilterEdge<<<gridSz1D, blkSz1D>>>(d_lSharp, d_l);
    nppiFilterGaussBorder_16s_C1R(d_lSharp, 1440 * 2, {1440, 1440}, {0, 0}, d_lSharp, 1440 * 2, {1440, 1440}, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE);
    EdgeAdd<<<gridSz1D, blkSz1D>>>(d_lSharp, d_l);
}

__global__ void bcsKer(uchar3 *lab, short *l, Params *param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((param->bright_post!=0)&&(param->contrast_post!=0))
    {
        lab[i].x = (uchar)clamp((float(l[i]) - 128) * param->contrast_post + 128 + param->bright_post, 0.0f, 255.0f);
    }
    else if(param->bright_post!=0)
    {
        lab[i].x = (uchar)clamp((float(l[i]) - 128) * param->contrast + 128 + param->bright_post, 0.0f, 255.0f);
    }
    else if(param->contrast_post!=0)
    {
        lab[i].x = (uchar)clamp((float(l[i]) - 128) * param->contrast_post + 128 + param->bright, 0.0f, 255.0f);
    }
    else
    {
        lab[i].x =(uchar)clamp((float(l[i]) - 128) * param->contrast + 128 + param->bright, 0.0f, 255.0f);
    }
}

// __global__ void gamSatKer(uchar4 *rgb, Params *param)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     rgb[i].x = powf(rgb[i].x/255.0f, param->gam) * 255.0f;
//     rgb[i].y = powf(rgb[i].y/255.0f, param->gam) * 255.0f;
//     rgb[i].z = powf(rgb[i].z/255.0f, param->gam) * 255.0f;

//     float p = sqrt(0.299f * rgb[i].x * rgb[i].x + 0.587f * rgb[i].y * rgb[i].y + 0.114f * rgb[i].z * rgb[i].z);

//     rgb[i].x = (uchar)clamp(p + ((float)rgb[i].x - p) * param->chroma, 0.0f, 255.f);
//     rgb[i].y = (uchar)clamp(p + ((float)rgb[i].y - p) * param->chroma, 0.0f, 255.f);
//     rgb[i].z = (uchar)clamp(p + ((float)rgb[i].z - p) * param->chroma, 0.0f, 255.f);
// }

extern "C" float mempvhighPtr()
{
    cudaMemcpy(h_pvhigh, d_pv_high, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return (float)*h_pvhigh / 2073600.0f;
}

extern "C" float mempvpeakPtr()
{
    cudaMemcpy(h_pvpeak, d_pv_peak, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return (float)*h_pvpeak / 2025.0f;
}

__global__ void pvKernel_avg(short * gray_l, unsigned int * pvblock)//, unsigned int * pv_local)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int pv_local[1];
    if(threadIdx.x == 0)
    {
        pv_local[0] = 0;
    }
    __syncthreads();
    unsigned int temp = (unsigned int)gray_l[i];
    // if(i/1440 >= 240 && i/1440 < 1200 && i%1440 >= 240 && i%1440 < 1200)
        atomicAdd(pv_local, temp);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        pvblock[blockIdx.x] = pv_local[0];
    }
}

// __global__ void pvTest(unsigned int *pv_high)
// {
//     int i = threadIdx.x;
//     pv_high[i] = 101;
// }

__global__ void pvKernel_add(unsigned int *pvblock, unsigned int *pv_high)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 2025)
    {
        if(threadIdx.x == 0)
        {
            pv_high[0] = 0;
        }
        __syncthreads();
        atomicAdd(pv_high, pvblock[i]);
    }
}

__global__ void pvKernel_pkflag(short *gray_l, unsigned int *pv_pkadd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int pv_pklocal[1];
    if(threadIdx.x == 0)
    {
        pv_pklocal[0] = 0;
    }
    __syncthreads();
    atomicMax(pv_pklocal, (unsigned int)gray_l[i]);
    __syncthreads();
    if(threadIdx.x == 0)
    {
        pv_pkadd[blockIdx.x] = pv_pklocal[0];
    }
}
__global__ void reduce_max_pass1(const short *in, short *out, int n) {
    extern __shared__ short sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;

    short x1 = (i < n) ? in[i] : -1;
    short x2 = (i + blockDim.x < n) ? in[i + blockDim.x] : -1;
    sdata[tid] = max(x1, x2);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        volatile short* vsmem = sdata;
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 1]);
    }

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

__global__ void reduce_max_pass2(const short *in, short *out, int n) {
    extern __shared__ short sdata[];
    unsigned int tid = threadIdx.x;
    if (tid < n)
        sdata[tid] = in[tid];
    else
        sdata[tid] = -1;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        volatile short* vsmem = sdata;
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 32]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 16]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 8]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 4]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 2]);
        vsmem[tid] = max(vsmem[tid], vsmem[tid + 1]);
    }

    if (tid == 0)
        out[0] = sdata[0];
} 
__inline__ void avgPk()
{
     // printf("inside average peak value \n");
    if(h_par->aecalc != 0)
    {
        // pvTest<<<1,1>>>(d_pv_high);
        // printf("ALC ON");
        if(h_par->aecalc == 1)
        {
            // printf("Average");
            pvKernel_avg<<<gridSz1D, blkSz1D>>>(d_l, d_pvblock);//, d_pv_high);
            pvKernel_add<<<(gridSz1D.x + blkSz1D.x - 1) / blkSz1D.x, blkSz1D>>>(d_pvblock, d_pv_high);
        }
        else
        {
            int blocks = ((imgWid * imgHei) + 1024*2-1) / (1024 *2);
            reduce_max_pass1<<<blocks, 1024, 1024 * sizeof(short)>>>(d_l, d_tmp, imgWid*imgHei);
            reduce_max_pass2<<<1, 1024, 1024 * sizeof(short)>>>(d_tmp, d_pv_peak, blocks);
            pvKernel_pkflag<<<gridSz1D, blkSz1D>>>(d_l, d_pv_pkadd);
            pvKernel_add<<<(gridSz1D.x + blkSz1D.x - 1) / blkSz1D.x, blkSz1D>>>(d_pv_pkadd, d_pv_peak);
        }
    }
    // printf("After average peak function execited\n");
}
//////////////////////////////////////vector Scope////////////////////////////
__global__ void postRoiVectorscope(unsigned char *cb, unsigned char *cr, unsigned int *op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int indx = (240 - int(cr[i])) * 225 + int(cb[i] - 16);
    op[indx] = 0x0000FF00;
}
__inline__ void vectorScope()
{
    cudaMemcpy(d_opChart, d_chart, 225 * 225 * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    nppiResize_8u_C4R((Npp8u *)d_init, hd_width * 4, {hd_width, hd_height}, {(hd_width - 640)/2, (hd_height - 480)/2, 640, 480}, (Npp8u *)d_postRoi, 64 * 4, {64, 48}, {0, 0, 64, 48}, 2);
    nppiFilterGaussBorder_8u_C4R((Npp8u *)d_postRoi, 64 * 4, {64, 48}, {0, 0}, (Npp8u *)d_postRoi, 64 * 4, {64, 48}, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE);
    nppiRGBToYCbCr_8u_AC4P3R((Npp8u *)d_postRoi, 64 * 4, planarYCbCr, 64, {64, 48});
    postRoiVectorscope<<<3, 1024>>>(planarYCbCr[1], planarYCbCr[2], d_opChart);

    cudaMemcpy(cbcrchart.data, d_opChart, 225 * 225 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(chart_color.data, d_postRoi, 64 * 48 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}
extern "C" unsigned char * download_vec_main()
{
    return cbcrchart.data;
}

extern "C" unsigned char * download_vec_color()
{
    return chart_color.data;
}

__global__ void planeshift_mblu1(uchar4 *rgb, int p1, float *d_plane1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(p1==0)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*1);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*1);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*1);
    }
    if(p1==1)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.9);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.9);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.7);
    }
    if(p1==2)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.6);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.8);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.6);
    }
    if(p1==3)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.9);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.8);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.8);
    }
    if(p1==4)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.8);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*1);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*1);
    }
    if(p1==5)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.7);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.8);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.9);
    }
    if(p1==6)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.7);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.6);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.8);
    }
    if(p1==7)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane1[9*p1+0] + (float)rgb[i].y * d_plane1[9*p1+1] + (float)rgb[i].z * d_plane1[9*p1+2])*0.4);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane1[9*p1+3] + (float)rgb[i].y * d_plane1[9*p1+4] + (float)rgb[i].z * d_plane1[9*p1+5])*0.4);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane1[9*p1+6] + (float)rgb[i].y * d_plane1[9*p1+7] + (float)rgb[i].z * d_plane1[9*p1+8])*0.3);
    }
}

__global__ void planeshift_mblu2(uchar4 *rgb, int p2, float *d_plane2)
{
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(p2==0)
    {
    rgb[i].x = (uchar)(((float)rgb[i].x * d_plane2[9*p2+0] + (float)rgb[i].y * d_plane2[9*p2+1] + (float)rgb[i].z * d_plane2[9*p2+2])*1);
    rgb[i].y = (uchar)(((float)rgb[i].x * d_plane2[9*p2+3] + (float)rgb[i].y * d_plane2[9*p2+4] + (float)rgb[i].z * d_plane2[9*p2+5])*1);
    rgb[i].z = (uchar)(((float)rgb[i].x * d_plane2[9*p2+6] + (float)rgb[i].y * d_plane2[9*p2+7] + (float)rgb[i].z * d_plane2[9*p2+8])*1);
    }
    if(p2==1)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane2[9*p2+0] + (float)rgb[i].y * d_plane2[9*p2+1] + (float)rgb[i].z * d_plane2[9*p2+2])*0.4);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane2[9*p2+3] + (float)rgb[i].y * d_plane2[9*p2+4] + (float)rgb[i].z * d_plane2[9*p2+5])*0.4);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane2[9*p2+6] + (float)rgb[i].y * d_plane2[9*p2+7] + (float)rgb[i].z * d_plane2[9*p2+8])*0.3);
    }
    if(p2==2)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane2[9*p2+0] + (float)rgb[i].y * d_plane2[9*p2+1] + (float)rgb[i].z * d_plane2[9*p2+2])*0.6);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane2[9*p2+3] + (float)rgb[i].y * d_plane2[9*p2+4] + (float)rgb[i].z * d_plane2[9*p2+5])*0.6);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane2[9*p2+6] + (float)rgb[i].y * d_plane2[9*p2+7] + (float)rgb[i].z * d_plane2[9*p2+8])*0.6);
    }
    if(p2==3)
    {
        rgb[i].x = (uchar)(((float)rgb[i].x * d_plane2[9*p2+0] + (float)rgb[i].y * d_plane2[9*p2+1] + (float)rgb[i].z * d_plane2[9*p2+2])*0.7);
        rgb[i].y = (uchar)(((float)rgb[i].x * d_plane2[9*p2+3] + (float)rgb[i].y * d_plane2[9*p2+4] + (float)rgb[i].z * d_plane2[9*p2+5])*0.7);
        rgb[i].z = (uchar)(((float)rgb[i].x * d_plane2[9*p2+6] + (float)rgb[i].y * d_plane2[9*p2+7] + (float)rgb[i].z * d_plane2[9*p2+8])*0.7);
    }

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void rgbGain(uchar4 *rgb, Params *param, float *wBal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float r = (float)rgb[i].x, g = (float)rgb[i].y, b = (float)rgb[i].z;

    float p = sqrt(0.299f * r * r + 0.587f * g * g + 0.114f * b * b);

    r = p + (r - p) * param->chroma;
    g = p + (g - p) * param->chroma;
    b = p + (b - p) * param->chroma;

    r = r * param->rGain;
    g = g * param->gGain;
    b = b * param->bGain;
    if(param->urGain!=0)
    {
        r = r * param->urGain;
    }
    if(param->ugGain!=0)
    {
        g = g * param->ugGain;
    }
    if(param->ubGain!=0)
    {
        b = b * param->ubGain;
    }
    if(param->wbFlag == 0)
    {
        r = (wBal[0] * r + wBal[1] * g + wBal[2] * b) * param->wbGain;
        g = (wBal[3] * r + wBal[4] * g + wBal[5] * b) * param->wbGain;
        b = (wBal[6] * r + wBal[7] * g + wBal[8] * b) * param->wbGain;
    }
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);

    r = powf(r/255.0f, param->gam) * 255.0f;
    g = powf(g/255.0f, param->gam) * 255.0f;
    b = powf(b/255.0f, param->gam) * 255.0f;

    rgb[i].x = (uchar)clamp(r, 0.0f, 255.0f);
    rgb[i].y = (uchar)clamp(g, 0.0f, 255.0f);
    rgb[i].z = (uchar)clamp(b, 0.0f, 255.0f);

}
__global__ void satKer(uchar4 *rgb, float satn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float p = sqrt(0.299f * rgb[i].x * rgb[i].x + 0.587f * rgb[i].y * rgb[i].y + 0.114f * rgb[i].z * rgb[i].z);

    rgb[i].x = (uchar)clamp(p + ((float)rgb[i].x - p) * satn, 0.0f, 255.f);
    rgb[i].y = (uchar)clamp(p + ((float)rgb[i].y - p) * satn, 0.0f, 255.f);
    rgb[i].z = (uchar)clamp(p + ((float)rgb[i].z - p) * satn, 0.0f, 255.f);
}
__global__ void replaceL1(short *l)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    l[i] = 100;
}
__inline__ void applyCLAHE(short *arr, float clipLim)
{
    printf("Inside apply claheeeeeeeeeeeeeee\n");
    // replaceL1<<<2025,1024>>>(arr);
    clahe->setClipLimit(clipLim);
    cudaMemcpy(clIn.data, arr, hd_width * hd_height * sizeof(short), cudaMemcpyDeviceToHost);
    clahe->apply(clIn, clOut);
    cudaMemcpy(arr, clOut.data, hd_width * hd_height * sizeof(short), cudaMemcpyHostToDevice);
    printf("Applied claheeeeeeeeeeeeeee\n");
}

__inline__ void imgBCS(float b, float c, float s, int p1, int p2)
{
    if(h_par->mblu == 1)
    {
        printf("\n Inside mblue 1 plane shift");
        planeshift_mblu1<<<gridSz1D, blkSz1D>>>((uchar4 *)d_init, p1, d_plane1);
    }
    if(h_par->mblu == 2)
    {
        printf("\n Inside mblue 2 plane shift");
        planeshift_mblu2<<<gridSz1D, blkSz1D>>>((uchar4 *)d_init, p2, d_plane2);
    }
    c4Toc3<<<gridSz1D, blkSz1D>>>((uchar4 *)d_init, (uchar3 *)d_init3);  //rgb to bgr

    nppiBGRToLab_8u_C3R(d_init3, hd_width * hd_channels, d_init3, hd_width * hd_channels, {hd_width, hd_height});

    sharpenYplane(h_par->sharp_value);

    // Apply clahe if m_blu is selected
    if(h_par->mblu > 0)
        applyCLAHE(d_l, h_par->hue_angle);

    if(h_par->mblu == 2)
        h_par->bright_post=h_par->bright_post+20;

    bcsKer<<<gridSz1D, blkSz1D>>>((uchar3 *)d_init3, d_l, d_param);
    avgPk();

    nppiLabToBGR_8u_C3R(d_init3, hd_width * 3, d_init3, hd_width* 3, {hd_width, hd_height});

    c3Toc4<<<gridSz1D, blkSz1D>>>((uchar3 *)d_init3, (uchar4 *)d_init);  //bgrtorgb
    satKer<<<gridSz1D, blkSz1D>>>( (uchar4 *)d_init, s);
    rgbGain<<<gridSz1D, blkSz1D>>>((uchar4 *)d_init, d_param, dWb);
}
__global__ void gamma_function(uchar4* rgb, float gamma)
{
    float l = 0.0f, h = 255.0f, d = h-l;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    rgb[i].x = clamp(rgb[i].x, l,h);
    rgb[i].y = clamp(rgb[i].y, l,h);
    rgb[i].z = clamp(rgb[i].z, l,h);


    rgb[i].x = powf((rgb[i].x - l) /d, gamma) * 255.0f;
    rgb[i].y = powf((rgb[i].y - l) /d, gamma) * 255.0f;
    rgb[i].z = powf((rgb[i].z - l) /d, gamma) * 255.0f;

    rgb[i].x = clamp(rgb[i].x, 0.0f, 255.0f);
    rgb[i].y = clamp(rgb[i].y, 0.0f, 255.0f);
    rgb[i].z = clamp(rgb[i].z, 0.0f, 255.0f);

    /*rgb[i].x = (unsigned char)clamp((powf(((float)rgb[i].x - 16) / 255.0f, gamma) * 255.0f), 0, 255);
    rgb[i].y = (unsigned char)clamp((powf(((float)rgb[i].y - 16) / 255.0f, gamma) * 255.0f), 0, 255);
    rgb[i].z = (unsigned char)clamp((powf(((float)rgb[i].z - 16) / 255.0f, gamma) * 255.0f), 0, 255);*/

}
void gamma_main(uchar4* rgb, float gamma)
{
    gamma_function << <gridSz1D,blkSz1D>> > (rgb, gamma);
}
__global__ void d_resize(cudaTextureObject_t tex, uint *out, float *d_zoomVal, Params *param)
{
    uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
    float u = x / (float) param->imageW;
    float v = y / (float) param->imageH;
    float bitmax = 255;
    float zoom = 1.0f/d_zoomVal[(int)param->zoom];
    float mv = (1- zoom)/2;
    float a = 1, b = 0;
    if(param->imageW == 1440)
    {
        a = 0.75f; b = 0.125;
    }
    float mv1 = mv + (b * zoom);
    v = v * zoom * a + mv1 + ((param->up - param->down) * mv1) / 1440;
    // else
    //     v = v * zoom + mv + ((param->up - param->down)*mv)/1440;
    u = u * zoom + mv + ((param->left - param->right) * mv) / 1440;
    // read from 3D texture
    float4 voxel = tex3D<float4>(tex, u, v, 0);
    if ((x < param->imageW) && (y < param->imageH))
    {
        // write output color
        uint i = (y * param->imageW) + x;
        out[i] = int(voxel.z * bitmax) << 16 | int(voxel.y * bitmax) << 8 | int(voxel.x * bitmax);
        // out[i] = 255<<8;
    }
}

__inline__ void resizeZoom()
{
    copyParams.srcPtr = make_cudaPitchedPtr((void *)d_init, volumeSize.width*sizeof(uchar4), volumeSize.width, volumeSize.height);
    cudaMemcpy3D(&copyParams);
    cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);
    rSzgrid = dim3((h_par->imageW + blockSize.x - 1) / blockSize.x, (h_par->imageH + blockSize.y - 1)/ blockSize.y);
    d_resize<<<rSzgrid, blockSize>>>(tex, d_out, d_zoomVal, d_param);
}

__global__ void mask(unsigned int * din, Params *param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ps, ns;
    float x2, y2, x0, y0, r0, r2;
    uchar maskType = param->maskType;
    int hei = param->imageH, wid = param->imageW;
    float fac = 1.0f, l = param->left * fac, r = param->right * fac, u = param->up * fac, d = param->down * fac, tri = param->triangle * fac;
    float wb2 = wid * 0.5f, hb2 = hei * 0.5f, x = i % wid, y = j % hei;

    if(maskType == 0)
    {
        if((x > (wid - r)) || (x <= l) || (y > (hei - d)) || (y <= u))
            din[j*wid+i] =  0x00000000;
    }
    else if(maskType == 1)
    {
        ps = y - x;
        ns = y + x;
        if((x > wid - r) || (x <= l) || (y > hei - d) || (y <= u) || (ps < tri - wid + r + u) || (ps >= hei - tri - l - d) || (ns < tri + l + u) || (ns >= hei - tri + wid - r - d))
            din[j*wid+i] =  0x00000000;
    }
    else if(maskType == 2)
    {
        x0 = wb2 + (l - r) / 2;
        y0 = hb2 + (u - d) / 2;
        r0 = wb2 - (l + r) / 2;
        x2 = (x - x0) * (x - x0);
        y2 = (y - y0) * (y - y0);
        r2 = r0 * r0;
        if((x2 + y2 > r2) || (y > hei - d) || (y <= u))
            din[j*wid+i] = 0;
    }
}
__inline__ void applyMask()
{
    maskgridSz = dim3((h_par->imageW + blockSize.x - 1) / blockSize.x, (h_par->imageH + blockSize.y - 1) / blockSize.y);
    mask<<<maskgridSz, blockSize>>>(d_out, d_param);
}
extern "C" void wbCpy(float *wb)
{
    cudaMemcpy(dWb, wb, sizeof(float) * 9, cudaMemcpyHostToDevice);
}
extern "C" unsigned int * imagePipeline2(short *h_rawImg)
{
    h_par = (Params *)&h_rawImg[2];

    cudaMemcpy(d_rawImg, h_rawImg, h_par->sensorH * h_par->sensorW * sizeof(ushort), cudaMemcpyHostToDevice);
    
    d_param = (Params *)&d_rawImg[2];
    static int count = 0;
    raw2cfa();

    // Black Level Correction
    blc <<<gridSize2D, blockSize2D >> >((ushort *)d_cfa, (ushort *)d_cfa, 10, imgHei, imgWid);

    //// Lens Shading Correction
    // lscdistancebased <<<gridSz1D, blkSz1D >> > ((ushort*) d_cfa, (ushort*) d_cfa, (float*)d_white_matrix, imgHei, imgWid);
    // lsctilebased<<<gridSz1D, blkSz1D >> > ((ushort*) d_cfa, (ushort*) d_cfa, (float*)d_white_matrix, imgHei, imgWid);

    // HDR Merging
    bits_extractor << < gridSize2D, blockSize2D >> > ((ushort*)d_cfa, (uchar*)lsb_img_8UC1, (uchar*)msb_img_8UC1, (uchar*)csb_img_8UC1, hd_width, hd_height);
    HDR << < gridSize2D, blockSize2D >> > ((uchar*)lsb_img_8UC1, (uchar*)csb_img_8UC1, (uchar*)msb_img_8UC1, (float*) d_hdr_weights, (uchar*)d_merged_bayer_8UC1);

    // Debayering
    nppiCFAToRGB_8u_C1C3R(d_merged_bayer_8UC1, hd_width * sizeof(uchar), {hd_width, hd_height}, {0, 0, hd_width, hd_height}, d_init8, hd_width * hd_channels * sizeof(uchar), NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED);
    c3Toc4copy<<<gridSz1D, blkSz1D>>>((uchar3 *)d_init8, (uchar4 *)d_init);

    // nppiCFAToRGB_16u_C1C3R(d_cfa, hd_width * sizeof(ushort), {hd_width, hd_height}, {0, 0, hd_width, hd_height}, d_init16, hd_width * hd_channels * sizeof(ushort), NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED);
    // convertUShort3ToInt<<<gridSz1D, blkSz1D>>>((ushort3*)d_init16, (uint*)d_init, hd_width, hd_height);

    //// Global Tone Mapping
    // if(h_par->wbFlag == 0)
    //     gtm(); // rgb output image

    // White Balancing
    if(h_par->wbFlag == 1)
        wbMain((uint *)d_init, dWb);

    // Image Color Adjustment
    imgBCS(h_par->bright_post, h_par->contrast_post, h_par->saturation_post,h_par->rSelect, h_par->gSelect);

    if(h_par->vec_scope)
        vectorScope();
    
    // Tone Mapping
    gamma_main((uchar4*)d_init, 0.65);

    // Resizing
    resizeZoom();

    // Masking for
    applyMask();

    // c3Toc4(d_merged_bayer_8UC1,);
    // c3Toc4copy<<<gridSz1D, blkSz1D>>>((uchar3 *)d_init8, (uchar4 *)d_init);  //rgbtorgb

    // return
    cudaMemcpy(h_out, d_out, hd_width * hd_height * sizeof(uchar4), cudaMemcpyDeviceToHost);
    return h_out;
}
