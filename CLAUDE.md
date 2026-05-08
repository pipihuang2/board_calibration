# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This directory is a camera calibration asset folder within a deep learning project. It currently contains a single circle-grid calibration board image generated from [calib.io](https://calib.io).

## Calibration Image

**File:** `calib.io_circles_200x150_8x11_12_5_01.bmp`

Filename encodes the pattern parameters:
- `circles` — asymmetric or symmetric circle grid (as opposed to chessboard)
- `200x150` — physical board dimensions in mm (width × height)
- `8x11` — grid layout (8 columns × 11 rows of circle centers)
- `12_5` — circle spacing/diameter in mm (12.5 mm)
- `01` — index/variant number

This image is typically used with OpenCV's `findCirclesGrid` function for intrinsic/extrinsic camera calibration. The circle grid pattern is preferred over chessboard when sub-pixel accuracy is critical, as circle centroid detection is more robust to noise.
