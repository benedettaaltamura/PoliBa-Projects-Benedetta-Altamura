#include "immagine.h"

// Definizioni di variabili globali
double PI = 3.14159265358979323846;
vector<vector<ImageType::IndexType>> seed_itk;
ImageType::IndexType seed_itk1;
ImageType::Pointer image_itk;
ImageTypeF::Pointer image_itkF;
ImageType::Pointer mask;


vector<ImageType::Pointer> itkImages;
vector<ImageTypeF::Pointer> itkImagesF;
vector<vector<Point>> seed_images;


bool compareByArea(const ComponentInfo& a, const ComponentInfo& b) {
	return a.area > b.area; 
}



void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		UserData* userData = static_cast<UserData*>(userdata);

		if (!userData->seedClicked) // Controlla se il primo clic è già stato fatto
		{
			cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;

			userData->seedPoint1 = Point(x, y);
			userData->seedClicked = true; // Imposta il flag a vero
			// Accedi all'immagine dall'oggetto userData
			const Mat& check_image = *(userData->opencvImage);
			// Ottieni il valore del pixel puntato
			unsigned char pixelValue = check_image.at<unsigned char>(y, x);
			cout << "Valore pixel: " << (int)pixelValue << endl;
		}
		else // Se il primo clic è stato fatto, registra il secondo punto di seme
		{
			cout << "Left button of the mouse is clicked (second seed) - position (" << x << ", " << y << ")" << endl;

			userData->seedPoint2 = Point(x, y);
			userData->seedClicked = false; // Imposta il flag a falso
			// Accedi all'immagine dall'oggetto userData
			const Mat& check_image = *(userData->opencvImage);
			// Ottieni il valore del pixel puntato
			unsigned char pixelValue = check_image.at<unsigned char>(y, x);
			cout << "Valore pixel: " << (int)pixelValue << endl;
		}
	}
}

void saveImageWithIndex(vector<Mat> image, const string& outputPath) {

	string main_output_folder = "C:\\Users\\bened\\OneDrive\\Desktop\\prova";
	// Salvataggio 
	for (int i = 0; i < image.size(); i++)
	{
		// Salvataggio dell'immagine con i bordi individuati in una cartella specifica
		string output_folder = main_output_folder + "\\image_" + to_string(i);
		string output_filename = output_folder + "\\" + outputPath + ".bmp";
		imwrite(output_filename, image[i]);
	}
}


vector<Mat> processImagesAndResize(const string& base_path)
{

	vector<Mat> image_list;  // Creazione di un vettore di oggetti cv::Mat per immagazzinare le immagini.
	vector<int> class_list;  // Creazione di un vettore di interi per immagazzinare le classi delle immagini.

	int index = 0; // Inizializzazione dell'indice per tenere traccia delle classi.
	int target_width = 300;  // Larghezza target per le immagini ridimensionate.
	int target_height = 300;  // Altezza target per le immagini ridimensionate.

	int totalWidthSum = 0; // Somma delle larghezze delle immagini.
	int totalHeightSum = 0; // Somma delle altezze delle immagini.

	for (const auto& class_path : fs::directory_iterator(base_path)) {
		// Itera attraverso le sottodirectory presenti nel percorso base.
		class_list.push_back(index);  // Aggiunge l'indice della classe al vettore delle classi.

		for (const auto& entry : fs::directory_iterator(class_path.path())) {
			// Itera attraverso i file presenti nella sottodirectory corrente.
			string filename = entry.path().string();  // Ottiene il percorso completo del file.

			Mat immagineOriginale = imread(filename); // La creo solo per verificare che l'immagine esista ed eseguire il controllo 
			if (immagineOriginale.data == NULL) {
				cerr << "Errore nell'apertura dell'immagine" << endl;
			}
			else {
				totalWidthSum += immagineOriginale.cols;
				totalHeightSum += immagineOriginale.rows;

				if (immagineOriginale.rows != target_width && immagineOriginale.cols != target_height) {
					// Modifico le dimensioni
					Mat resized_image;
					resize(immagineOriginale, resized_image, Size(target_width, target_height));

					// Applico l'operatore di Canny all'immagine ridimensionata
					Mat edges;
					Mat edgesX;
					Mat edgesY;
					//Canny(resized_image, edges, 100, 200);  //Soglie messe a caso
					//Sobel
					//Sobel(resized_image, edgesX, -1, 1, 0, 3, BORDER_DEFAULT); //sobel x
					//Sobel(resized_image, edgesY, -1, 0, 1, 3, BORDER_DEFAULT); //sobel y
					//for (int i = 0; i < edgesX.rows; i++) {
					//	for (int j = 0; j < edgesX.cols; j++) {
					//		edges.at<unsigned char>(i,j) = sqrt(pow(edgesX.at<unsigned char>(i,j), 2) + pow(edgesY.at<unsigned char>(i,j), 2));
					//	}
					//}

					// Salvataggio dell'immagine con i bordi individuati in una cartella specifica
					/*string output_folder = "C:\\Users\\marti\\Downloads\\New\\";
					string output_filename = output_folder + fs::path(filename).filename().string();
					imwrite(output_filename, edges);*/
				}
				image_list.push_back(imread(filename));  // Ottiene il percorso completo del file.
			}
		}

		index++;  // Aumenta l'indice della classe.
	}

	int totalImages = image_list.size();
	double averageWidth = static_cast<double>(totalWidthSum) / totalImages;
	double averageHeight = static_cast<double>(totalHeightSum) / totalImages;

	cout << "Dimensioni medie delle immagini originali - Larghezza: " << averageWidth << ", Altezza: " << averageHeight << endl;
	cout << "Numero totale di immagini: " << totalImages << endl;
	cout << "Numero totale di classi: " << class_list.size() << endl;
	return image_list;
}

// Step 1) 
vector<Mat> normalizzazione(vector<Mat> image) {
	for (int w = 0; w < image.size(); w++)
	{
		//calcolo valore massimo, minimo e medio
		float min = 255;
		float max = 0;
		float mean = 0;
		for (int i = 0; i < image[w].rows; i++)
		{
			for (int j = 0; j < image[w].cols; j++)
			{
				if (image[w].at<uchar>(i, j) < min)
				{
					min = image[w].at<uchar>(i, j);
				}
				if (image[w].at<uchar>(i, j) > max)
				{
					max = image[w].at<uchar>(i, j);
				}
				mean += image[w].at<uchar>(i, j);
			}
		}
		cout << "il pixel minimo e': " << min << " il pixel massimo e': " << max << endl;

		mean = float((mean) / (image[w].rows * image[w].cols));
		//cout << " Il valore massimo dell'immagine " << w << " è: " << max << "  " << "Il valore minimo è : " << min << "  " << "La media è : " << mean << endl;

		// Equalizzo
		int norm = 0;
		for (int i = 0; i < image[w].rows; i++)
		{
			for (int j = 0; j < image[w].cols; j++)
			{
				norm = (int)(((image[w].at<uchar>(i, j) - min) / (max - min)) * 255);

				if (norm > 255)
					norm = 255;
				if (norm < 0)
					norm = 0;
				image[w].at<uchar>(i, j) = norm;
				//printf("Pixel [%d][%d]: Grayscale value=%u\n", i, j, mat[i][j]);
			}

		}
	}
	return image;

} //Step 1 

// Step 2) 
vector<Mat> binarizzazione_media_no_nero(vector<Mat> images) {
	//APPLICO LA SOGLIA
	vector<Mat> bin_Image;

	bin_Image.resize(images.size());
	for (int w = 0; w < images.size(); w++)
	{
		// Calcolo la media 
		float media = 0;
		for (int i = 0; i < images[w].rows; i++)
		{
			for (int j = 0; j < images[w].cols; j++)
			{
				media += images[w].at<uchar>(i, j);

			}

		}
		media = (float)media / (images[w].rows * images[w].cols);

		//CALCOLO DEV ST

		double stdDev = 0.0;
		for (int i = 0; i < images[w].rows; i++)
		{
			for (int j = 0; j < images[w].cols; j++)
			{
				stdDev += pow((images[w].at<uchar>(i, j)) - media, 2);

			}

		}

		stdDev = sqrt(stdDev / (images[w].rows * images[w].cols));
		cout << "media" << media << " e dev std: " << stdDev << endl;

		// Imposta la soglia per ottenere sfondo nero e resto bianco
		double thresholdValue = media + 2 * stdDev;  // Valore di soglia
		double max_value = 255; // Valore massimo assegnato ai pixel sopra la soglia, corrisponde al bianco
		double min_value = 0;
		for (int i = 0; i < images[w].rows; i++)
		{
			for (int j = 0; j < images[w].cols; j++)
			{
				if (images[w].at<uchar>(i, j) < min_value)
				{
					min_value = images[w].at<uchar>(i, j);
				}
				if (images[w].at<uchar>(i, j) > max_value)
				{
					max_value = images[w].at<uchar>(i, j);
				}

			}
		}


		//scegli il tipo di threshold:
		//threshold(images[w], bin_Image[w], thresholdValue, max_value, THRESH_BINARY);
		//threshold(images[w], bin_Image[w], min_value, max_value, THRESH_BINARY | THRESH_OTSU);
		//adaptiveThreshold(images[w], bin_Image[w], max_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, 0); 
		//adaptiveThreshold(images[w], bin_Image[w], max_value, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 0);
		threshold(images[w], bin_Image[w], thresholdValue, max_value, THRESH_BINARY);
		Mat concat;
		hconcat(images[w], bin_Image[w], concat);
		//imshow("treshold", concat);
		//waitKey(0);

	}


	return bin_Image;
}

vector<Mat> threshold_function(vector<Mat>images)
{
	//APPLICO LA SOGLIA
	vector<Mat> thresholdedImage;

	thresholdedImage.resize(images.size());
	for (int w = 0; w < images.size(); w++)
	{
		// Calcolo la media 
		float media = 0;
		for (int i = 0; i < images[w].rows; i++)
		{
			for (int j = 0; j < images[w].cols; j++)
			{
				media += images[w].at<uchar>(i, j);
			}

		}
		media = (float)media / (images[w].rows * images[w].cols);
		//cout << "Media n°" << w << " e: " << media << endl;

		// Imposta la soglia per ottenere sfondo nero e resto bianco
		double thresholdValue = media; // Valore di soglia
		double maxValue = 255; // Valore massimo assegnato ai pixel sopra la soglia, corrisponde al bianco
		double minValue = 0;

		//scegli il tipo di threshold:
		//threshold(images[w], thresholdedImage[w], thresholdValue, maxValue, THRESH_BINARY);
		//threshold(images[w], thresholdedImage[w], minValue, maxValue, THRESH_BINARY | THRESH_OTSU);
		//adaptiveThreshold(images[w], thresholdedImage[w], maxValue, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, 0); 
		//adaptiveThreshold(images[w], thresholdedImage[w], maxValue, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 0);
		threshold(images[w], thresholdedImage[w], minValue, maxValue, THRESH_BINARY | THRESH_OTSU);

		//cv::imshow("Otsu Threshold", thresholdedImage[w]);
		//cv::waitKey(0);

	}
	return thresholdedImage;
}

// Step 3) 5)
vector<Mat> regioni_maggiori(vector<Mat>images, vector<Mat>images_originale, int componenti) {

	vector<Mat> bigger_image;
	bigger_image.resize(images.size());

	for (int w = 0; w < images.size(); w++) {

		Mat labels, stats, centroids;
		int numLabels = connectedComponentsWithStats(images[w], labels, stats, centroids);
		vector<ComponentInfo> componentList; //vettore di strutture 

		for (int i = 1; i < numLabels; ++i)
		{
			ComponentInfo info;
			info.label = i;
			info.area = stats.at<int>(i, cv::CC_STAT_AREA);
			componentList.push_back(info);
		}

		// Ordina il vettore componentList in base all'area in ordine decrescente utilizzando la funzione di confronto compareByArea.
		// Ovvero prima la regione con area maggiore
		sort(componentList.begin(), componentList.end(), compareByArea);

		Mat outputImage = Mat::zeros(images[w].size(), CV_8U);

		// Copia le regioni delle prime componenti componenti (quindi le più grandi) nell'immagine outputImage. 
		// Questo viene fatto utilizzando  le etichette delle componenti e creando una maschera per ogni componente.

		// Ciclo for viene utilizzato per scorrere attraverso le prime 2 componenti all'interno della componentList
		for (int i = 0; i < min(componenti, static_cast<int>(componentList.size())); ++i)
		{
			//  Per ogni componente, viene creata una maschera utilizzando le etichette delle componenti 
			// e l'immagine outputImage viene aggiornata assegnando il valore 255 ai pixel corrispondenti nella maschera.
			int label = componentList[i].label;
			Mat mask = (labels == label);
			outputImage.setTo(255, mask); // Assegna 255 solo ai pixel corrispondenti a mask true
		}
		// Visualizza l'immagine originale e l'immagine sogliata
		/*Mat combined_images;
		Mat combined_images1;
		hconcat(images_originale[w], images[w], combined_images);
		hconcat(combined_images, outputImage, combined_images1);
		imshow("Binarizzata con threshold prima e dopo", combined_images1);
		waitKey(0);*/

		bigger_image[w] = outputImage;
		//imshow("Regione Maggiore dell'immagine", bigger_image[w]);
		//waitKey(0);
	}
	return bigger_image;
}

// Step 4)
vector<Mat> erosione(vector<Mat> images)
{
	vector<Mat> eroded_image;
	eroded_image.resize(images.size());
	for (int w = 0; w < images.size(); w++)
	{
		// Definisci l'elemento strutturale per l'erosione
		// MORPH_ELLIPSE (ellisse) è la forma dell'elemento strutturante
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		erode(images[w], eroded_image[w], kernel, Point(-1, -1));

		//Mat combined_images;
		//hconcat(images[w], eroded_image[w], combined_images);
		//imshow("Immagine erosa", combined_images);
		//waitKey(0);
	}

	return eroded_image;
}

// Step 6)
vector<Mat> dilatazione(vector<Mat> images, vector<Mat> image_original)
{
	vector<Mat> dilated_image, filled_image;
	dilated_image.resize(images.size());
	filled_image.resize(images.size());

	for (int w = 0; w < images.size(); w++)
	{
		// Definisci l'elemento strutturale per l'erosione
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		dilate(images[w], dilated_image[w], kernel, Point(-1, -1));
		//chiusura
		morphologyEx(dilated_image[w], filled_image[w], MORPH_CLOSE, kernel, Point(-1, -1), 5);

		//Mat combined_images;
		//hconcat(image_original[w], dilated_image[w], combined_images);
		//Mat combined_images1;
		//hconcat(combined_images, filled_image[w], combined_images1);
		//imshow("binarizzazionne-dilatazione-chiusura", combined_images1);
		//waitKey(0);
	}

	return filled_image;
}

// Step 7)
vector<Mat> maschera(vector<Mat> mask, vector<Mat> image_original)
{
	vector<Mat> image_mask;
	image_mask.resize(mask.size());
	for (int w = 0; w < mask.size(); w++) {
		// Crea la maschera dall'immagine chiusa
		// Applica la maschera all'immagine originale
		Mat resultImage;
		image_original[w].copyTo(image_mask[w], mask[w]);
		//imshow("immagine con maschera", image_mask[w]);
		//waitKey(0);
	}
	return image_mask;

}

// Step 8)
vector<Mat> filtri(vector<Mat> image_list_raw)
{
	// 1) FILTRO SHMOOTING
	vector<Mat>  filteredImage;
	filteredImage.resize(size(image_list_raw)); //Devono avere le stesse dimensioni

	for (int w = 0; w < image_list_raw.size(); w++)
	{
		//medianBlur(image_list_raw[w], filteredImage[w], 3);// Applica il filtro mediano con una finestra 3x3  //filtro che ha come elementi: 1/righe*colonne
		GaussianBlur(image_list_raw[w], filteredImage[w], Size(3, 3), 0);
		//imshow("Original Image", image_list_raw[w]);
		//imshow("Filtered Image", filteredImage[w]);
		//waitKey(0);
	}



	// 2) FILTRO DI SHARPENING, 
	// Definizione del kernel di sharpening Laplaciano
	Mat laplacianKernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);

	// Applicare la convoluzione con il kernel di sharpening Laplaciano
	vector<Mat> sharpened;
	vector<Mat> sharpenedAbs;
	sharpened = filteredImage; //Devono avere le stesse dimensioni
	sharpenedAbs = filteredImage; //Devono avere le stesse dimensioni
	for (int w = 0; w < filteredImage.size(); w++)
	{
		filter2D(filteredImage[w], sharpened[w], CV_32F, laplacianKernel);
		// Converti l'immagine risultante in un'immagine visualizzabile
		convertScaleAbs(sharpened[w], sharpenedAbs[w]);
		Mat concatenazione;
		/*hconcat(image_list_raw[w], sharpenedAbs[w], concatenazione);
		imshow("pre e post filtri", concatenazione);

		waitKey(0);*/

	}
	return sharpenedAbs;
}



void trasformata_hough_c(vector<Mat> image) {
	// Applica la trasformata di Hough circolare
	for (int w = 0; w < image.size(); w++)
	{
		Mat segmented_image = Mat::zeros(image[w].rows, image[w].cols, CV_8UC1);;
		Mat edgesX;
		Mat edgesY;

		Canny(image[w], segmented_image, 150, 180, 3);
		Mat segmented_image_color;
		cvtColor(segmented_image, segmented_image_color, COLOR_GRAY2BGR);
		imshow("canny image", segmented_image);
		waitKey(0);
		vector<Vec3f> circles;
		Mat color_image;
		cvtColor(image[w], color_image, COLOR_GRAY2BGR);
		HoughCircles(segmented_image, circles, HOUGH_GRADIENT, 1, 50, 180, 30, 3, 50);
		// Disegna i cerchi individuati sull'immagine originale
		for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// circle center
			circle(color_image, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
			// circle outline
			int radius = c[2];
			circle(color_image, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
		}
		Mat concat;
		hconcat(segmented_image_color, color_image, concat);
		imshow("detected circles", concat);
		waitKey(0);


		if (circles.size() == 1) {
			Vec3i c = circles[0];
			Point center = Point(c[0], c[1]);
			seed_images.push_back(vector<Point>{center, Point(0, 0)});
		}
		else {
			seed_images.push_back(vector<Point>{Point(0, 0), Point(0, 0)});
		}


		cv::Point selectedPoint = seed_images[w][0];
		int x = selectedPoint.x;
		int y = selectedPoint.y;
		ImageType::IndexType p1;
		p1[0] = x;
		p1[1] = y;

		cv::Point selectedPoint1 = seed_images[w][1];
		int x1 = selectedPoint1.x;
		int y1 = selectedPoint1.y;
		ImageType::IndexType p2;
		p2[0] = x1;
		p2[1] = y1;
		seed_itk.push_back(vector<ImageType::IndexType>{p1, p2});


	}

}

void seleziona_seed(vector<Mat> image) {

	for (int w = 0; w < image.size(); w++)
	{
		UserData userData;
		userData.opencvImage = &image[w];
		userData.seedClicked = false; // Inizialmente nessun clic è stato fatto
		userData.seedPoint1 = Point(0, 0); // Inizializza il primo seedPoint a (0,0)
		userData.seedPoint2 = Point(0, 0); // Inizializza il secondo seedPoint a (0,0)

		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", CallBackFunc, &userData);

		//show the image
		imshow("My Window", *userData.opencvImage);

		// Wait until user press some key
		waitKey(0);
		if (userData.seedClicked) // Controlla se il secondo clic è stato fatto 
		{
			// Aggiungi un punto di seme (o due punti uguali) al vettore seed_images
			Point p = userData.seedPoint1;
			seed_images.push_back(vector<Point>{p, p});

		}
		else
		{
			// Aggiungi entrambi i punti di seme al vettore seed_images
			Point p1 = userData.seedPoint1;
			Point p2 = userData.seedPoint2;
			seed_images.push_back(vector<Point>{p1, p2});

		}

		cv::Point selectedPoint = seed_images[w][0];
		int x = selectedPoint.x;
		int y = selectedPoint.y;
		ImageType::IndexType p1;
		p1[0] = x;
		p1[1] = y;

		cv::Point selectedPoint1 = seed_images[w][1];
		int x1 = selectedPoint1.x;
		int y1 = selectedPoint1.y;
		ImageType::IndexType p2;
		p2[0] = x1;
		p2[1] = y1;
		seed_itk.push_back(vector<ImageType::IndexType>{p1, p2});



		// Stampa i valori delle coordinate seed_itk e seed_itk1
		//cout << "seed_itk: (" << seed_itk[w][0] << ", " << seed_itk[w][1] << ")" << endl; 


	}
}



// Salvo le coordinate del seed in un file testo
void saveSeedImagesToFile(const vector<PointVector>& seed_images, const string& filename)
{
	ofstream outputFile(filename, ios::out);

	if (!outputFile.is_open())
	{
		cout << "Errore nell'apertura del file." << endl;
		return;
	}

	for (const PointVector& seed_image : seed_images)
	{
		for (const Point& point : seed_image)
		{
			outputFile << point.x << " " << point.y << " ";
		}
		outputFile << endl;
	}

	outputFile.close();
}
// Decomprimo il file testo
void loadSeedImagesFromFile(vector<PointVector>& seed_images, const string& filename)
{
	ifstream inputFile(filename);

	if (!inputFile.is_open())
	{
		cout << "Errore nell'apertura del file." << endl;
		return;
	}

	seed_images.clear();
	string line;
	while (getline(inputFile, line))
	{
		istringstream iss(line);
		PointVector points;
		int x, y;
		while (iss >> x >> y)
		{
			points.push_back(Point(x, y));
		}
		seed_images.push_back(points);

		cv::Point selectedPoint = points[0];
		int c = selectedPoint.x;
		int r = selectedPoint.y;
		ImageType::IndexType p1;
		p1[0] = c;
		p1[1] = r;

		cv::Point selectedPoint1 = points[1];
		int c1 = selectedPoint1.x;
		int r1 = selectedPoint1.y;
		ImageType::IndexType p2;
		p2[0] = c1;
		p2[1] = r1;
		seed_itk.push_back(vector<ImageType::IndexType>{p1, p2});

	}

	inputFile.close();
}

void opencv2itk(vector<Mat> image) {

	for (const cv::Mat& opencvImage : image) {

		ImageType::Pointer itkImage = ImageType::New();
		ImageTypeF::Pointer itkImageF = ImageTypeF::New();

		ImageType::IndexType start;
		start.Fill(0);

		ImageTypeF::IndexType startF;
		startF.Fill(0);

		ImageType::SizeType size;
		size[0] = opencvImage.cols;
		size[1] = opencvImage.rows;

		ImageTypeF::SizeType sizeF;
		sizeF[0] = opencvImage.cols;
		sizeF[1] = opencvImage.rows;

		ImageType::RegionType region;
		region.SetIndex(start);
		region.SetSize(size);

		ImageTypeF::RegionType regionF;
		regionF.SetIndex(startF);
		regionF.SetSize(sizeF);

		itkImage->SetRegions(region);
		itkImage->Allocate();

		itkImageF->SetRegions(regionF);
		itkImageF->Allocate();

		for (int y = 0; y < opencvImage.rows; y++) {
			for (int x = 0; x < opencvImage.cols; x++) {
				unsigned char pixelValue = opencvImage.at<unsigned char>(y, x);
				itkImage->SetPixel({ x, y }, pixelValue);

				float pixelValueF = opencvImage.at<unsigned char>(y, x);
				itkImageF->SetPixel({ x, y }, pixelValueF);
			}
		}

		itkImages.push_back(itkImage);
		itkImagesF.push_back(itkImageF);

	}
}


//REGION GROWING PER TROVARE LA LESIONE 


Mat RegionGrowing(Mat sourceImage, Point seed, double threshold) {

	int counter = 0;

	// LISTA PUNTI DA ESPLORARE
	list<Point> listaPunti; 

	//definiamo ora la maschera che all'inizio è vuota
	Mat region = Mat::zeros(sourceImage.size(), sourceImage.type());

	//ora aggiungiamo il seed alla lista di punti per cui va esplorato il vicinato
	listaPunti.push_back(seed); 

	//Oltre ad aggiungerlo alla lista io aggiungo il seed alla regione stessa. 
	region.at<uchar>(seed) = 255; 

	//Ora esploro la lista dei punti, valuto il vicinato (a 4) di ogni punto partendo dal seed
	Point neighbours[4];
	Mat maskImage; //Immagine temporanea
	Mat blendedImage;
	Mat image;

	//Per verificare se la lista sia vuota utilizzo il metodo empty
	while (!listaPunti.empty())
	{
		counter = counter + 1;
		Point next = listaPunti.front(); //front ritorna il primo elemento della lista
		
		// Esploriamo la lista a partire dal primo punto e nel momento in cui troviamo dei punti
		// Li aggiungiamo dietro
		//esploriamo il punto subito a sinistra del punto di partenza
		neighbours[0].x = next.x - 1;
		neighbours[0].y = next.y;

		//facciamo la stessa cosa con il punto subito a dx
		neighbours[1].x = next.x + 1;
		neighbours[1].y = next.y;

		//SOTTO
		neighbours[2].x = next.x;
		neighbours[2].y = next.y - 1;
		//SOPRA
		neighbours[3].x = next.x;
		neighbours[3].y = next.y + 1;

		// utilizziamo il metodo inside per capire se un Point è incluso all'interno di un rect di dimensioni pari alla mia immagine.
		Rect range = Rect(0, 0, sourceImage.cols, sourceImage.rows);
		double regionMean = regionMeanValue(sourceImage, region);


		//Per ogni vicino del vicinato del punto di partenza 
		for (int i = 0; i < 4; i++)
		{
			// se il vicino i-esimo è un punto dell'immagine:
			if (neighbours[i].inside(range))
			{
				if (region.at<uchar>(neighbours[i]) == 0)
				{
					//Questa differenca va presa col valore assoluto  e il double serve per utilizzare il uchar
					double dist = abs((double)sourceImage.at<uchar>(neighbours[i]) - regionMean) / 255;
					// se il punto appartiene già alla maschera significa che è stato già esplorato!
					if (dist < threshold)
					{
						listaPunti.push_back(neighbours[i]);
						region.at<uchar>(neighbours[i]) = 255;
					}
				}
			}
		}

		// Elimino il punto appena investigato dalla lista
		listaPunti.pop_front();

		if (counter % 1200 == 0)
		{
			// Mostrare la crescita della regione
			cv::cvtColor(region, maskImage, COLOR_GRAY2BGR);
			cv::cvtColor(sourceImage, image, COLOR_GRAY2BGR);
			addWeighted(image, 0.5, maskImage, 0.5, 0.0, blendedImage);
			imshow("RegionGrowing", blendedImage);
			waitKey(1);
		}

	}

	return region;
}

double regionMeanValue(Mat sourceImage, Mat mask) {  // restituisce il valor medio della regione segmentata
	double mean = 0.0;
	int counter = 0;

	for (int i = 0; i < sourceImage.rows; i++)
	{
		for (int j = 0; j < sourceImage.cols; j++)
		{
			if (mask.at<uchar>(i, j) > 0)
			{
				mean = mean + sourceImage.at<uchar>(i, j);
				counter = counter + 1;
			}
		}
	}
	mean = mean / counter;
	return mean;
}

Mat watershedOpencv(Mat sourceImage, Mat imm_orig, int w) {

	Mat rgb_image;
	cvtColor(sourceImage, rgb_image, COLOR_GRAY2RGB);

	Mat rgb_imm_orig;
	cvtColor(imm_orig, rgb_imm_orig, COLOR_GRAY2RGB);

	//per evitare la sovrasegmentazione dò i seed a partire dai quali fa la segmentazione
	Mat dist = Mat::zeros(sourceImage.size(), CV_8U);
	for (int i = 0; i < dist.rows; i++) {
		for (int j = 0; j < dist.cols; j++) {
			if (Point(j, i) == seed_images[w][0] || Point(j, i) == seed_images[w][1]) {
				dist.at<uchar>(i, j) = 255;
				dist.at<uchar>(i + 1, j) = 255;
				dist.at<uchar>(i - 1, j) = 255;
				dist.at<uchar>(i, j + 1) = 255;
				dist.at<uchar>(i, j - 1) = 255;
				dist.at<uchar>(i - 1, j - 1) = 255;
				dist.at<uchar>(i - 1, j + 1) = 255;
				dist.at<uchar>(i + 1, j - 1) = 255;
				dist.at<uchar>(i + 1, j + 1) = 255;
			}
		}
	}


	vector<vector<Point>> contours;
	findContours(dist, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//cout<< "dimensione: " << contours.size() << endl;

	Mat markers = Mat::zeros(dist.size(), CV_32S);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
	}

	watershed(rgb_image, markers);

	//trovo controrni dopo watershed
	Mat mark;
	markers.convertTo(mark, CV_8U);

	int borderMargin = 5; // Margine attorno ai bordi dell'immagine

	for (int i = borderMargin; i < mark.rows - borderMargin; i++)
	{
		for (int j = borderMargin; j < mark.cols - borderMargin; j++)
		{
			if (mark.at<uchar>(i, j) != 0) {
				mark.at<uchar>(i, j) = 0;
			}
			else {
				mark.at<uchar>(i, j) = 255;
			}
		}
	}

	// Imposta tutti i pixel del bordo a 0
	for (int i = 0; i < mark.rows; i++)
	{
		for (int j = 0; j < mark.cols; j++)
		{
			if (i < borderMargin || i >= mark.rows - borderMargin ||
				j < borderMargin || j >= mark.cols - borderMargin)
			{
				mark.at<uchar>(i, j) = 0;
			}
		}
	}

	vector<vector<Point>> contoursM;
	findContours(mark, contoursM, RETR_CCOMP, CHAIN_APPROX_NONE);
	drawContours(mark, contoursM, 1, 255, -1);

	Mat dst = mark;
	Mat concat;
	hconcat(imm_orig, dst, concat);
	imshow("Final Result", concat);
	waitKey(0);

	return dst;
}


//region growing per trovare la lesione con itk
ImageType::Pointer RegionGrowing_isolated_connected(ImageTypeF::Pointer image, int w) {
	// Filtro 2 che eseguirà il processo di segmentazione

	bool mode;

	if (image->GetPixel(seed_itk[w][0]) > image->GetPixel(seed_itk[w][1]))
	{
		cout << "Pixel seed 1 ha intensita' maggiore del pixel seed 2" << endl;
		mode = true;//		FindUpperThresholdOff: Quando è off allora la modalità è quella per separare regioni chiare circondate da regioni scure 
	}
	else
	{
		cout << "Pixel seed 1 ha intensita' minore del pixel seed 2" << endl;
		mode = false;//		FindUpperThresholdOn : Quando è on allora la modalità è quella per separare regioni scure circondate da regioni chiare trovando automaticamente una soglia superiore di isolamento minima
	}

	using ConnectedFilterType = itk::IsolatedConnectedImageFilter< ImageTypeF, ImageType >;
	ConnectedFilterType::Pointer isolatedConnected = ConnectedFilterType::New();



	isolatedConnected->SetInput(image);

	isolatedConnected->SetLower(100); //Usato come criterio per determinare quali pixel appartengono alla regione segmentata
	isolatedConnected->SetUpper(255);
	isolatedConnected->AddSeed1(seed_itk[w][0]);
	isolatedConnected->AddSeed2(seed_itk[w][1]);
	isolatedConnected->SetReplaceValue(255);

	if (mode)
	{
		isolatedConnected->FindUpperThresholdOff();
	}
	else {
		isolatedConnected->FindUpperThresholdOn();
	}
	isolatedConnected->GetIsolatedValue();
	std::cout << "Valore isolato: " << isolatedConnected->GetIsolatedValue() << std::endl;
	isolatedConnected->Update(); 

	// Verifica se l'algoritmo di segmentazione è riuscito a trovare la soglia di isolamento
	bool thresholdingFailed = isolatedConnected->GetThresholdingFailed();
	if (thresholdingFailed) {
		cout << "Thresholding failed. No isolating threshold found." << endl;
	}
	else {
		cout << "Thresholding successful. Isolating threshold found." << endl;
	}

	ImageType::Pointer mask = isolatedConnected->GetOutput(); // Ottiene il puntatore all'immagine di output

	return mask;
}

ImageType::Pointer RegionGrowing_confidence_connected(ImageType::Pointer image, int w) {

	//USO IL METODO CONFIDENCE CONNECTED
	using ConnectedFilterType = itk::ConfidenceConnectedImageFilter< ImageType, ImageType >;
	ConnectedFilterType::Pointer confidenceConnected = ConnectedFilterType::New();
	confidenceConnected->SetMultiplier(2.5);
	confidenceConnected->SetNumberOfIterations(1);
	confidenceConnected->SetReplaceValue(255);
	confidenceConnected->SetSeed(seed_itk[w][0]);
	confidenceConnected->SetInitialNeighborhoodRadius(3);
	confidenceConnected->SetInput(image);
	confidenceConnected->Update();
	ImageType::Pointer mask = confidenceConnected->GetOutput();
	ImageType::RegionType inputRegion = mask->GetLargestPossibleRegion();
	ImageType::SizeType regionSize = inputRegion.GetSize();


	return mask;
}

//segmentazione automatica in base a intensità dei pixel 
Mat segmentazione_aut(Mat binImage, Mat sourceImage, Mat imm_orig) {

	Mat dist;
	distanceTransform(binImage, dist, DIST_L2, 3);

	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	//imshow("Distance Transform Image", dist);
	//waitKey(0);


	threshold(dist, dist, 0.2, 1.0, THRESH_BINARY);
	//imshow("Distance Transform Image", dist);
	//waitKey(0);

	Mat kernel1 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel1);

	Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(dist, dist, MORPH_CLOSE, kernel2, Point(-1, -1), 5);
	//imshow("Distance Transform Image", dist);
	//waitKey(0);


	//8u image serve per findContours()
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point>> contours;
	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


	//creo immagine nera con regioni trovate in bianco
	Mat markers = Mat::zeros(dist.size(), CV_32S);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), 255, -1);
	}

	Mat markers8u;
	markers.convertTo(markers8u, CV_8U, 10);
	imshow("", markers8u);
	waitKey(0);


	return markers8u;
}

///////////////////LAVORO SULLA LESIONE///////////////////////

//trovo momenti usando le funzioni di opencv
void openCVMoments(vector<Mat> image_list)
{
	for (int i = 0; i < image_list.size(); i++)
	{

		Mat image = image_list[i];
		double humm[7];
		Moments moment = moments(image, false);
		HuMoments(moment, humm);

		//cout << "OpenCV Raw Moments:\t" << moment.m00 << "\t" << moment.m01 << "\t" << moment.m10 << "\t"
		//	<< moment.m11 << "\t" << moment.m02 << "\t" << moment.m20 <<
		//	"\t" << moment.m12 << "\t" << moment.m21 << "\t" << moment.m03 << "\t" << moment.m30 << endl;

		//cout << "OpenCV Central Moments:\t" << moment.mu11 << "\t" << moment.mu02 << "\t" << moment.mu20 <<
		//	"\t" << moment.mu12 << "\t" << moment.mu21 << "\t" << moment.mu03 << "\t" << moment.mu30 << endl;

		//cout << "OpenCV Norm. Central Moments:\t" << moment.nu11 << "\t" << moment.nu02 << "\t" << moment.nu20 <<
		//	"\t" << moment.nu12 << "\t" << moment.nu21 << "\t" << moment.nu03 << "\t" << moment.nu30 << endl;

		cout << "OpenCV Hu Moments:\t" << humm[0] << "\t" << humm[1] << "\t" << humm[2] <<
			"\t" << humm[3] << "\t" << humm[4] << "\t" << humm[5] << "\t" << humm[6] << endl;
	}
}
//trovo momenti senza funzioni opencv
void customMoments(vector<Mat> image)
{
	for (int w = 0; w < image.size(); w++) {

		// Passo 1: Calcolare i momenti semplici (raw moments)
		double m00 = 0.0, m01 = 0.0, m10 = 0.0,
			m11 = 0.0, m02 = 0.0, m20 = 0.0,
			m30 = 0.0, m03 = 0.0, m21 = 0.0, m12 = 0.0;

		for (int i = 0; i < image[w].rows; i++)
			for (int j = 0; j < image[w].cols; j++)
			{
				m00 += (double)image[w].at<unsigned char>(i, j);
				m01 += (double)i * image[w].at<unsigned char>(i, j);
				m10 += (double)j * image[w].at<unsigned char>(i, j);
				m11 += (double)i * j * image[w].at<unsigned char>(i, j);
				m02 += (double)i * i * image[w].at<unsigned char>(i, j);
				m20 += (double)j * j * image[w].at<unsigned char>(i, j);
				m12 += (double)i * i * j * image[w].at<unsigned char>(i, j);
				m21 += (double)i * j * j * image[w].at<unsigned char>(i, j);
				m03 += (double)i * i * i * image[w].at<unsigned char>(i, j);
				m30 += (double)j * j * j * image[w].at<unsigned char>(i, j);
			}



		//cout << "Custom Raw Moments:\t" << m00 << "\t" << m01 << "\t" << m10 << "\t"
		//	<< m11 << "\t" << m02 << "\t" << m20 << 
		//	"\t" << m12 << "\t" << m21 << "\t" << m03 << "\t" << m30 << endl;



		// Passo 2: Calcolo i Momenti Centrali
		double mu00 = 0.0, mu01 = 0.0, mu10 = 0.0,
			mu11 = 0.0, mu02 = 0.0, mu20 = 0.0,
			mu30 = 0.0, mu03 = 0.0, mu21 = 0.0, mu12 = 0.0;



		// Centrodi
		cout << "Centroid coordinates " << endl;
		double x_sign = 0.0, y_sign = 0.0;
		x_sign = m10 / m00;
		y_sign = m01 / m00;
		cout << " X coordinate : " << x_sign << "\t" << endl;
		cout << " Y coordinate : " << y_sign << "\t" << endl;



		// By definition
		mu00 = m00;
		mu01 = 0;
		mu10 = 0;


		for (int i = 0; i < image[w].rows; i++)
			for (int j = 0; j < image[w].cols; j++)
			{
				double i_val = (i - y_sign);
				double j_val = (j - x_sign);



				mu11 += (double)i_val * j_val * image[w].at<unsigned char>(i, j);
				mu02 += (double)i_val * i_val * image[w].at<unsigned char>(i, j);
				mu20 += (double)j_val * j_val * image[w].at<unsigned char>(i, j);
				mu12 += (double)i_val * i_val * j_val * image[w].at<unsigned char>(i, j);
				mu21 += (double)i_val * j_val * j_val * image[w].at<unsigned char>(i, j);
				mu03 += (double)i_val * i_val * i_val * image[w].at<unsigned char>(i, j);
				mu30 += (double)j_val * j_val * j_val * image[w].at<unsigned char>(i, j);
			}

		cout << "Custom Central Moments:\t" << endl;
		cout << " Moment mu11 : " << mu11 << "\t" << endl;
		cout << " Moment mu02 : " << mu02 << "\t" << endl;
		cout << " Moment mu20 : " << mu20 << "\t" << endl;
		cout << " Moment mu12 : " << mu12 << "\t" << endl;
		cout << " Moment mu21 : " << mu21 << "\t" << endl;
		cout << " Moment mu03 : " << mu03 << "\t" << endl;
		cout << " Moment mu30 : " << mu30 << "\t" << endl;

		// Passo 3: Calcolo dei Momenti Centrali Normalizzati
		double ni11 = 0.0, ni02 = 0.0, ni20 = 0.0,
			ni30 = 0.0, ni03 = 0.0, ni21 = 0.0, ni12 = 0.0;



		ni11 = mu11 / pow(m00, 2);
		ni02 = mu02 / pow(m00, 2);
		ni20 = mu20 / pow(m00, 2);
		ni30 = mu30 / pow(m00, 2.5);
		ni03 = mu03 / pow(m00, 2.5);
		ni21 = mu21 / pow(m00, 2.5);
		ni12 = mu12 / pow(m00, 2.5);



		cout << "Custom Norm. Central Moments:\t" << endl;
		cout << " Moment ni11 : " << ni11 << "\t" << endl;
		cout << " Moment ni02 : " << ni02 << "\t" << endl;
		cout << " Moment ni20 : " << ni20 << "\t" << endl;
		cout << " Moment ni12 : " << ni12 << "\t" << endl;
		cout << " Moment ni21 : " << ni21 << "\t" << endl;
		cout << " Moment ni03 : " << ni03 << "\t" << endl;
		cout << " Moment ni30 : " << ni30 << "\t" << endl;



		// Passo 4: Hu Moments
		double I1 = 0.0, I2 = 0.0, I3 = 0.0, I4 = 0.0, I5 = 0.0, I6 = 0.0, I7 = 0.0;
		I1 = ni20 + ni02;
		I2 = pow((ni20 - ni02), 2) + 4 * ni11 * ni11;
		I3 = pow((ni30 - 3 * ni12), 2) + pow((3 * ni21 - ni03), 2);
		I4 = pow((ni30 + ni12), 2) + pow((ni21 + ni03), 2);
		I5 = (ni30 - 3 * ni12) * (ni30 + ni12) * (pow((ni30 + ni12), 2) - 3 * pow((ni21 + ni03), 2)) +
			(3 * ni21 - ni03) * (ni21 + ni03) * (3 * pow((ni30 + ni12), 2) - pow((ni21 + ni03), 2));
		I6 = (ni20 - ni02) * (pow((ni30 + ni12), 2) - pow((ni21 + ni03), 2)) + 4 * ni11 * (ni30 + ni12) * (ni21 + ni03);
		I7 = (3 * ni21 - ni03) * (ni30 + ni12) * (pow((ni30 + ni12), 2) - 3 * pow((ni21 + ni03), 2)) -
			(ni30 - 3 * ni12) * (ni21 + ni03) * (3 * pow((ni30 + ni12), 2) - pow((ni21 + ni03), 2));



		//cout << "Custom Hu Moments image n:\t" << w << " : " << I1 << "\t" << I2 << "\t" << I3 <<
		//	"\t" << I4 << "\t" << I5 << "\t" << I6 << "\t" << I7 << endl;
		cout << "Custom Hu Moments image:\t" << endl;
		cout << " Moment 1 : " << I1 << "\t" << endl;
		cout << " Moment 2 : " << I2 << "\t" << endl;
		cout << " Moment 3 : " << I3 << "\t" << endl;
		cout << " Moment 4 : " << I4 << "\t" << endl;
		cout << " Moment 5 : " << I5 << "\t" << endl;
		cout << " Moment 6 : " << I6 << "\t" << endl;
		cout << " Moment 7 : " << I7 << "\t" << endl;



		// Aggiunta del vettore di centroidi alla lista totale dei centroidi
		Mat color_image;
		cvtColor(image[w], color_image, COLOR_GRAY2RGB);
		circle(color_image, Point(x_sign, y_sign), 10, Scalar(0, 0, 255));
		imshow("Image with circle", color_image);
		waitKey(0);



	}
}

void centroidCoordinates(vector<Mat> image_list)
{
	vector<Point> centroids;
	for (int w = 0; w < image_list.size(); w++)
	{
		// Passo 1: Calcolare i momenti semplici (raw moments)
		double m00 = 0.0, m01 = 0.0, m10 = 0.0;
		for (int i = 0; i < image_list[w].rows; i++)
			for (int j = 0; j < image_list[w].cols; j++)
			{
				m00 += (double)image_list[w].at<uchar>(i, j);
				m01 += (double)i * image_list[w].at<uchar>(i, j);
				m10 += (double)j * image_list[w].at<uchar>(i, j);
			}

		// Centrodi
		double x_sign = 0.0, y_sign = 0.0;
		x_sign = m10 / m00;
		y_sign = m01 / m00;

		// Creazione di un vettore di double per rappresentare il centroide
		vector<Point> centroids;
		centroids.push_back(Point(x_sign, y_sign));



		// Aggiunta del vettore di centroidi alla lista totale dei centroidi
		Mat color_image;
		cvtColor(image_list[w], color_image, COLOR_GRAY2RGB);
		circle(color_image, Point(x_sign, y_sign), 10, Scalar(0, 0, 255));
		imshow("Image with circle", color_image);
		waitKey(0);
	}

}

void axisInfo(vector<Mat> image)
{
	for (int w = 0; w < image.size(); w++)
	{
		cout << "Informazioni immagine n° " << w << endl;

		double m00 = 0.0, m01 = 0.0, m10 = 0.0;

		for (int i = 0; i < image[w].rows; i++)
			for (int j = 0; j < image[w].cols; j++)
			{
				m00 += static_cast<double>(image[w].at<uchar>(i, j));
				m01 += static_cast<double>(i * image[w].at<uchar>(i, j));
				m10 += static_cast<double>(j * image[w].at<uchar>(i, j));
			}

		//cout << "Custom Raw Moments:\t" << m00 << "\t" << m01 << "\t" << m10 << "\t"
		//	<< m11 << "\t" << m02 << "\t" << m20 << 
		//	"\t" << m12 << "\t" << m21 << "\t" << m03 << "\t" << m30 << endl;

		// Passo 2: Calcolo i Momenti Centrali
		double mu00 = 0.0, mu01 = 0.0, mu10 = 0.0,
			mu11 = 0.0, mu02 = 0.0, mu20 = 0.0;

		// Centrodi
		double x_sign = 0.0, y_sign = 0.0;
		x_sign = m10 / m00;
		y_sign = m01 / m00;

		// By definition
		mu00 = m00;
		mu01 = 0;
		mu10 = 0;


		for (int i = 0; i < image[w].rows; i++)
			for (int j = 0; j < image[w].cols; j++)
			{
				double i_val = (i - y_sign);
				double j_val = (j - x_sign);

				mu11 += static_cast<double> (i_val * j_val * image[w].at<uchar>(i, j));
				mu02 += static_cast<double> (i_val * i_val * image[w].at<uchar>(i, j));
				mu20 += static_cast<double>(j_val * j_val * image[w].at<uchar>(i, j));
			}

		double I1 = (0.5 * (mu20 + mu02)) + (0.5 * sqrt(4 * pow(mu11, 2) + pow((mu20 - mu02), 2)));
		double I2 = (0.5 * (mu20 + mu02)) - (0.5 * sqrt(4 * pow(mu11, 2) + pow((mu20 - mu02), 2)));
		double theta = 0.5 * atan(2 * mu11 / (mu20 - mu02));

		double temp;
		if (mu20 < mu02)
		{
			temp = I1;
			I1 = I2;
			I2 = temp;
		}

		double a1 = 2 * sqrt(I1 / mu00);
		double a2 = 2 * sqrt(I2 / mu00);

		double angle = 0.0;
		angle = theta * 180 / PI;

		Mat image_color;
		cvtColor(image[w], image_color, COLOR_GRAY2RGB);
		ellipse(image_color, Point((int)x_sign, (int)y_sign), Size((int)a1, (int)a2), angle, 0, 360, Scalar(0, 0, 255));

		Point start, end, start1, end1;

		// NB: Cos e Sin vogliono l'angolo in radianti
		start = Point((int)(x_sign + a1 * cos(theta)), (int)(y_sign + a1 * sin(theta)));
		end = Point((int)(x_sign - a1 * cos(theta)), (int)(y_sign - a1 * sin(theta)));

		start1 = Point((int)(x_sign + a2 * sin(theta)), (int)(y_sign - a2 * cos(theta)));
		end1 = Point((int)(x_sign - a2 * sin(theta)), (int)(y_sign + a2 * cos(theta)));



		line(image_color, start, end, Scalar(0, 255, 0));

		line(image_color, start1, end1, Scalar(255, 0, 0));

		imshow("Ellipse Axis Image" + to_string(w), image_color);
		waitKey(0);

		if (a1 > a2)
		{
			// Calcolo dell'eccentricità
			double eccentricity = sqrt(1 - (pow(a2, 2) / pow(a1, 2)));
			// Stampa delle informazioni sull'asse maggiore, asse minore e eccentricità
			cout << "Asse maggiore: " << a1 << endl;
			cout << "Asse minore: " << a2 << endl;
			cout << "Eccentricita': " << eccentricity << endl;

		}
		else
		{
			double copia;
			copia = a1;
			a1 = a2;
			a2 = copia;
			// Calcolo dell'eccentricità
			double eccentricity = sqrt(1 - (pow(a2, 2) / pow(a1, 2)));
			// Stampa delle informazioni sull'asse maggiore, asse minore e eccentricità
			cout << "Asse maggiore: " << a1 << endl;
			cout << "Asse minore: " << a2 << endl;
			cout << "Eccentricita': " << eccentricity << endl;
		}

	}


}

void statistica(vector<Mat> image) {

	for (int w = 0; w < image.size(); w++) {

		//CALCOLO ISTOGRAMMA
		int channel = 0;
		int histSize = 256;
		float grayranges[] = { 0, 256 };
		const float* range[] = { grayranges };
		Mat hist;
		//
		calcHist(&image[w], 1, &channel, Mat(), hist, 1, &histSize, range, true, false);


		// Dimensioni immagine istogramma
		int hist_w = 512;
		int hist_h = 512;
		// Larghezza colonna singolo bin
		int bin_w = cvRound((double)hist_w / histSize);
		// Fattore di scala applicato a ogni bin -> utili solo per rappresentazione grafica
		int scale = 5;
		// Immagine contenente l'istogramma
		Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0));
		int max = image[w].rows * image[w].cols;

		for (int i = 0; i < histSize; i++)
		{
			// Estraggo i singoli valori dell'istograma e il scalo per poterli rappresentare
			int value = cvRound(hist.at<float>(i) * hist_h / max * scale);

			//cout << value << " ";
			rectangle(histImage,
				Point(i * bin_w, hist_h),
				Point(i * bin_w + bin_w, hist_h - value),
				Scalar::all(255),
				FILLED
			);
		}
		cout << endl;

		//namedWindow("Istogramma", WINDOW_AUTOSIZE);
		//imshow("Istogramma", histImage);
		//waitKey(0);


		//CALCOLO MEDIA 
		double numEle = 0;
		double sum = 0.0;
		for (int i = 0; i < hist.rows; i++)
		{
			sum = sum + i * hist.at<float>(i);
			numEle = numEle + hist.at<float>(i);
		}

		double media = sum / numEle;


		//CALCOLO DEV ST
		numEle = 0;
		double stdDev = 0.0;
		for (int i = 0; i < hist.rows; i++)
		{
			stdDev = stdDev + hist.at<float>(i) * pow((double)i - media, 2);
			numEle = numEle + hist.at<float>(i);
		}

		stdDev = sqrt(stdDev / numEle);


		//CALCOLO SKEWNESS
		numEle = 0;

		double moment3 = 0.0;
		double skew = 0.0;

		for (int i = 0; i < hist.rows; i++)
		{
			moment3 = moment3 + pow((double)i - media, 3) * hist.at<float>(i);
			numEle = numEle + hist.at<float>(i);
		}

		moment3 = moment3 / numEle;
		skew = moment3 / pow(stdDev, 3.0);

		//CALCOLO KURTOSIS
		numEle = 0;

		double moment4 = 0.0;
		double kurt = 0.0;

		for (int i = 0; i < hist.rows; i++)
		{
			moment4 = moment4 + pow((double)i - media, 4) * hist.at<float>(i);
			numEle = numEle + hist.at<float>(i);
		}

		moment4 = moment4 / numEle;
		kurt = moment4 / pow(stdDev, 4.0) - 3;

		cout << "statistica immagine n°: " << w + 1 << endl;
		cout << "media: " << media << endl;
		cout << "std: " << stdDev << endl;
		cout << "skew: " << skew << endl;
		cout << "kurt: " << kurt << endl;

	}


}

void descrittori_geometrici(vector<Mat> images, vector<int>area_dilated_image)
{
	// DESCRITTORI GEOMETRICI LESIONE

	for (int w = 0; w < images.size(); w++) {
		imshow("", images[w]);
		waitKey(0);
		cout << "IMMAGINE N°" << w << endl;
		// 1) Area della lesione
		double area = countNonZero(images[w]);
		cout << "  pixel della lesione : " << area << endl;
		area = (double)(area / area_dilated_image[w]);
		cout << "  Area della lesione: " << area << endl;

		// 2) Perimetro della lesione
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(images[w], contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);

		for (int c = 0; c < contours.size(); c++) {
			const auto& contour = contours[c];  //contour è un riferimento a un oggetto il cui tipo viene determinato automaticamente dal compilatore, ed è costante, il che significa che l'oggetto a cui fa riferimento non può essere modificato tramite questa variabile.
			double perimeter = arcLength(contour, true);
			cout << "  Perimetro del contorno : " << perimeter << endl;

			// 3) Indice di compattezza della lesione
			double contour_area = contourArea(contour);
			double compactness = 1 - ((4 * CV_PI * contour_area) / (perimeter * perimeter));
			cout << "  Compattezza del contorno : " << compactness << endl;

			// 4) Diametro del contorno
			double max_distance = 0.0;
			for (size_t i = 0; i < contour.size(); i++) {
				for (size_t j = i + 1; j < contour.size(); j++) {
					double distance = norm(contour[i] - contour[j]);
					if (distance > max_distance) {
						max_distance = distance;
					}
				}
			}
			cout << "  Diametro del contorno : " << max_distance << endl;
			cout << " " << endl;
		}
	}

}

/// POST-PROCESSING
vector<vector<vector<Point>>> post_processing(vector<Mat> mask_lesion, vector<Mat> filt_image)
{
	//11) Tappo i buchi della lesione
	for (size_t w = 0; w < mask_lesion.size(); ++w) {
		//chiusura
		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));  // Esempio: kernel rettangolare 5x5
		morphologyEx(mask_lesion[w], mask_lesion[w], MORPH_CLOSE, kernel, Point(-1, -1), 5);
		//imshow("tappo buchi", mask_lesion[w]);
		//waitKey(0);
	}

	// 12) EVIDENZIO LA REGIONE DI INTERESSE  SULL'IMMAGINE GRAYSCALE
	//IMMAGINE ORIGINALE CON LESIONE IN ROSSO 
	vector<Mat> coloredImages;
	// Ciclo attraverso le immagini e le maschere
	for (size_t w = 0; w < filt_image.size(); ++w) {
		cv::Mat coloredImage;

		// Applica la maschera per estrarre la regione di interesse
		cv::Mat roi;
		filt_image[w].copyTo(roi, mask_lesion[w]);

		// Crea un'immagine colorata copiando l'immagine originale
		cv::cvtColor(filt_image[w], coloredImage, cv::COLOR_GRAY2BGR);

		// Imposta il colore rosso (BGR: 0, 0, 255) nella regione di interesse
		coloredImage.setTo(cv::Scalar(0, 0, 255), mask_lesion[w]);

		coloredImages.push_back(coloredImage);
		// Visualizza l'immagine colorata
		//cv::imshow("Immagine colorata", coloredImage);
		//cv::waitKey(0);


	}

	vector<vector<vector<Point>>> contorni_lesione;
	//IMMAGINE ORIGINALE CON LESIONE CONTORNATA 
	vector<Mat>images_with_contours;
	for (int w = 0; w < mask_lesion.size(); w++)
	{
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mask_lesion[w], contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE); //I contorni sono identificati come sequenze di punti che formano il bordo delle regioni bianche nell'immagine.

		Mat image_with_contours;
		cvtColor(filt_image[w], image_with_contours, COLOR_GRAY2BGR);
		// 0.5 alla fine
		drawContours(image_with_contours, contours, -1, Scalar(0, 255, 0), 2.5);

		imshow("Image with Contours", image_with_contours);
		waitKey(0);

		images_with_contours.push_back(image_with_contours);;
		contorni_lesione.push_back(contours);
	}

	return contorni_lesione;

}



