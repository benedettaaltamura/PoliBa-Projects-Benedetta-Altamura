
#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageSeriesReader.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkIntensityWindowingImageFilter.h>
#include "itkOpenCVImageBridge.h"
#include <itkOrientImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkViewImage.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryMorphologicalClosingImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkConfidenceConnectedImageFilter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkImageSeriesWriter.h>
#include <itkImageToVTKImageFilter.h>
#include <vtkSmartPointer.h>
#include <vtkImageViewer2.h>
#include "vtkMarchingCubes.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "itkRegionOfInterestImageFilter.h"
#include "vtkSTLWriter.h"
#include "vtkMassProperties.h"
#include "itkCurvatureFlowImageFilter.h"
#include "vtkAutoInit.h"
#include "itkIsolatedConnectedImageFilter.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkCurvatureFlowImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkWatershedImageFilter.h"
#include "itkScalarToRGBColormapImageFilter.h"
#include "itkScalarToRGBPixelFunctor.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"




namespace fs = std::experimental::filesystem;
using namespace std;
using namespace cv;


const int Dimension = 2;

//dichiarazione alias per i tipi di dati 
using PixelType = unsigned char;
using PixelTypeF = float;
using ImageType = itk::Image<PixelType, Dimension>;
using ImageTypeF = itk::Image<PixelTypeF, Dimension>; 
using PointVector = vector<Point>;// Definisci il tipo di dati PointVector come alias per vector<Point>


// Dichiarazioni di variabili globali
extern const int Dimension; 
extern double PI; 
extern std::vector<std::vector<ImageType::IndexType>> seed_itk;   
extern ImageType::IndexType seed_itk1; 
extern ImageType::Pointer image_itk;
extern ImageTypeF::Pointer image_itkF;
extern ImageType::Pointer mask;
extern std::vector<ImageType::Pointer> itkImages;
extern std::vector<ImageTypeF::Pointer> itkImagesF;
extern std::vector<std::vector<cv::Point>> seed_images;


struct ComponentInfo {
	int label;
	int area;
};


///////////////////////////////// PARTE itk 
struct UserData {
	cv::Mat* opencvImage;
	bool seedClicked; // Flag per tracciare se è stato già fatto un clic sull'immagine
	Point seedPoint1;  // Primo punto di seme
	Point seedPoint2;  // Secondo punto di seme
};


//dichiarazioni funzioni 
bool compareByArea(const ComponentInfo& a, const ComponentInfo& b);
void saveImageWithIndex(vector<Mat>, const string&); 
vector<Mat> processImagesAndResize(const string&);
vector<Mat> normalizzazione(vector<Mat>);
vector<Mat> threshold_function(vector<Mat>);
vector<Mat> regioni_maggiori(vector<Mat>, vector<Mat>, int); //3
vector<Mat> erosione(vector<Mat>); //4
vector<Mat> dilatazione(vector<Mat>, vector<Mat>); //6
vector<Mat> maschera(vector<Mat>, vector<Mat>); //7
vector<Mat> filtri(vector<Mat>);
vector<Mat> binarizzazione_media_no_nero(vector<Mat>);
vector<vector<vector<Point>>> post_processing(vector<Mat>, vector<Mat>); 

void trasformata_hough_c(vector<Mat>); 
void seleziona_seed(vector<Mat>);
void saveSeedImagesToFile(const vector<PointVector>&, const string&);
void loadSeedImagesFromFile(vector<PointVector>&, const string&);


void opencv2itk(vector<Mat>);

void openCVMoments(vector<Mat>);
void customMoments(vector<Mat>);
void centroidCoordinates(vector<Mat>);
void axisInfo(vector<Mat>);
void statistica(vector<Mat>);
void descrittori_geometrici(vector<Mat>, vector<int>);



Mat RegionGrowing(Mat, Point, double);
double regionMeanValue(Mat, Mat);
Mat watershedOpencv(Mat, Mat,int);
Mat segmentazione_aut(Mat binImage, Mat sourceImage, Mat imm_orig); 
ImageType::Pointer RegionGrowing_isolated_connected(ImageTypeF::Pointer, int); 
ImageType::Pointer RegionGrowing_confidence_connected(ImageType::Pointer, int); 









//GESTIONE EVENTI DEL MOUSE
#if defined(__APPLE__) || defined(__MACH__)
#define PLATFORM_NAME "macos"
#else
#define PLATFORM_NAME "windows"
#endif





