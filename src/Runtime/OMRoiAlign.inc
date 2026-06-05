/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- OMRoiAlign.inc - OMTensor C/C++ neutral implementation -----===//
//
// Copyright 2026
//
// =============================================================================
//
// This file contains C/C++ neutral implementation for RoiAlign runtime helper.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "onnx-mlir/Runtime/OMTensor.h"

static float roiAlignBilinearInterpolate(
    const float *feature, int64_t height, int64_t width, float y, float x) {
  if (y < -1.f || y > (float)height || x < -1.f || x > (float)width)
    return 0.f;

  if (y <= 0.f)
    y = 0.f;
  if (x <= 0.f)
    x = 0.f;

  int64_t yLow = (int64_t)y;
  int64_t xLow = (int64_t)x;
  int64_t yHigh = 0;
  int64_t xHigh = 0;

  if (yLow >= height - 1) {
    yHigh = yLow = height - 1;
    y = (float)yLow;
  } else {
    yHigh = yLow + 1;
  }

  if (xLow >= width - 1) {
    xHigh = xLow = width - 1;
    x = (float)xLow;
  } else {
    xHigh = xLow + 1;
  }

  const float ly = y - (float)yLow;
  const float lx = x - (float)xLow;
  const float hy = 1.f - ly;
  const float hx = 1.f - lx;

  const float v1 = feature[yLow * width + xLow];
  const float v2 = feature[yLow * width + xHigh];
  const float v3 = feature[yHigh * width + xLow];
  const float v4 = feature[yHigh * width + xHigh];

  return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

void RoiAlign(OMTensor *output, OMTensor *X, OMTensor *rois,
    OMTensor *batchIndices, char *coordinateTransformationMode, char *mode,
    int64_t outputHeight, int64_t outputWidth, int64_t samplingRatio,
    double spatialScaleDouble) {
  assert(output && X && rois && batchIndices && "RoiAlign runtime: null input");
  assert(omTensorGetDataType(output) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(X) == ONNX_TYPE_FLOAT &&
         omTensorGetDataType(rois) == ONNX_TYPE_FLOAT &&
         "RoiAlign runtime: currently only float tensors are supported");
  assert(omTensorGetDataType(batchIndices) == ONNX_TYPE_INT64 &&
         "RoiAlign runtime: batchIndices must be int64");
  assert(omTensorGetRank(X) == 4 && "RoiAlign runtime: X rank must be 4");
  assert(omTensorGetRank(rois) == 2 && "RoiAlign runtime: rois rank must be 2");
  assert(omTensorGetRank(batchIndices) == 1 &&
         "RoiAlign runtime: batchIndices rank must be 1");
  assert(omTensorGetRank(output) == 4 &&
         "RoiAlign runtime: output rank must be 4");
  assert(strcmp(mode, "avg") == 0 &&
         "RoiAlign runtime: only mode=avg is supported");
  assert(strcmp(coordinateTransformationMode, "half_pixel") == 0 &&
         "RoiAlign runtime: only coordinate_transformation_mode=half_pixel is "
         "supported");

  const int64_t *xShape = omTensorGetShape(X);
  const int64_t *roisShape = omTensorGetShape(rois);
  const int64_t *outputShape = omTensorGetShape(output);
  const int64_t numRois = roisShape[0];
  const int64_t channels = xShape[1];
  const int64_t height = xShape[2];
  const int64_t width = xShape[3];
  (void)outputShape;

  assert(roisShape[1] == 4 && "RoiAlign runtime: rois second dim must be 4");
  assert(outputShape[0] == numRois && outputShape[1] == channels &&
         outputShape[2] == outputHeight && outputShape[3] == outputWidth &&
         "RoiAlign runtime: output shape mismatch");

  const float spatialScale = (float)spatialScaleDouble;
  const float offset = 0.5f;

  const float *xData = (const float *)omTensorGetDataPtr(X);
  const float *roisData = (const float *)omTensorGetDataPtr(rois);
  const int64_t *batchIndicesData = (const int64_t *)omTensorGetDataPtr(batchIndices);
  float *outData = (float *)omTensorGetDataPtr(output);

  const int64_t featureMapStride = channels * height * width;
  const int64_t channelStride = height * width;
  const int64_t outputRoiStride = channels * outputHeight * outputWidth;
  const int64_t outputChannelStride = outputHeight * outputWidth;

  for (int64_t n = 0; n < numRois; ++n) {
    const int64_t batchIndex = batchIndicesData[n];
    assert(batchIndex >= 0 && batchIndex < xShape[0] &&
           "RoiAlign runtime: invalid batch index");
    const float *roi = roisData + n * 4;

    const float roiStartW = roi[0] * spatialScale - offset;
    const float roiStartH = roi[1] * spatialScale - offset;
    const float roiEndW = roi[2] * spatialScale - offset;
    const float roiEndH = roi[3] * spatialScale - offset;

    const float roiWidth = roiEndW - roiStartW;
    const float roiHeight = roiEndH - roiStartH;
    const float binSizeH = roiHeight / (float)outputHeight;
    const float binSizeW = roiWidth / (float)outputWidth;

    int64_t roiBinGridH = samplingRatio > 0 ? samplingRatio
                                             : (int64_t)ceilf(roiHeight /
                                                              (float)outputHeight);
    int64_t roiBinGridW = samplingRatio > 0 ? samplingRatio
                                             : (int64_t)ceilf(roiWidth /
                                                              (float)outputWidth);
    if (roiBinGridH <= 0)
      roiBinGridH = 1;
    if (roiBinGridW <= 0)
      roiBinGridW = 1;

    const float count = (float)(roiBinGridH * roiBinGridW);
    const float *batchData = xData + batchIndex * featureMapStride;
    float *outputRoi = outData + n * outputRoiStride;

    for (int64_t c = 0; c < channels; ++c) {
      const float *feature = batchData + c * channelStride;
      float *outputChannel = outputRoi + c * outputChannelStride;
      for (int64_t ph = 0; ph < outputHeight; ++ph) {
        for (int64_t pw = 0; pw < outputWidth; ++pw) {
          float accum = 0.f;
          for (int64_t iy = 0; iy < roiBinGridH; ++iy) {
            const float y = roiStartH + (float)ph * binSizeH +
                            ((float)iy + 0.5f) * binSizeH / (float)roiBinGridH;
            for (int64_t ix = 0; ix < roiBinGridW; ++ix) {
              const float x = roiStartW + (float)pw * binSizeW +
                              ((float)ix + 0.5f) * binSizeW /
                                  (float)roiBinGridW;
              accum += roiAlignBilinearInterpolate(feature, height, width, y, x);
            }
          }
          outputChannel[ph * outputWidth + pw] = accum / count;
        }
      }
    }
  }
}
