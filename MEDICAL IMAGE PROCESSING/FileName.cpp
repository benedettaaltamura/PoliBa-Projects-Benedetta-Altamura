#include "immagine.h"




int main()
{
	
	vector<Mat> image_list;  // Creazione di un vettore di oggetti Mat per immagazzinare le immagini.
	vector<Mat> image_list_raw;  // Creazione di un vettore di oggetti Mat per immagazzinare le immagini raw.
	vector<int> class_list;  // Creazione di un vettore di interi per immagazzinare le classi delle immagini.
	string base_path = "C:\\Users\\marti\\Downloads\\DATASET BRAIN TUMOR MRI bmp"; // Directory

	int index = 0; // Inizializzazione dell'indice per tenere traccia delle classi.

	
	
	///////////////////LEGGO I DATI//////////////
	for (const auto& class_path : fs::directory_iterator(base_path)) {
		// Itera attraverso le sottodirectory presenti nel percorso base.
		class_list.push_back(index);  // Aggiunge l'indice della classe al vettore delle classi.

		for (const auto& entry : fs::directory_iterator(class_path.path())) {
			// Itera attraverso i file presenti nella sottodirectory corrente.
			string filename = entry.path().string();  // Ottiene il percorso completo del file.

			Mat immagineOriginale = imread(filename, IMREAD_GRAYSCALE); // La creo solo per verificare che l'immagine esista ed eseguire il controllo 
			//imshow("Image", immagineOriginale);
			//waitKey(0);


			if (immagineOriginale.data == NULL) {
				cerr << "Errore nell'apertura dell'immagine" << endl;
				return -1;
			}
			else {
				image_list_raw.push_back(immagineOriginale); //Salvo l'immagine grezza nel vettore
			}
		}

		index++;  // Aumenta l'indice della classe.
	}

	int totalImages = image_list.size();
	cout << "Numero totale di immagini: " << totalImages << endl;
	cout << "Numero totale di classi: " << class_list.size() << endl;

	


	////////////////////// INDIVIDUAZIONE AREA DEL CERVELLO ED ELIMINAZIONE DI SFONDO E OSSA ////////////////////////

	// 0) STATISTICHE IMMAGINI
	statistica(image_list_raw); 


	// 1) NORMALIZZO LE IMMAGINI GREZZE

	vector<Mat> normalized_image = normalizzazione(image_list_raw); // o normalizzazione
	//statistica(normalized_image);

	// 2) BINARIZZAZIONE con soglia pari alla media dei pixel
	//vector<Mat>binar_image = binarizzazione_media(normalized_image);  //metodo con media
	vector<Mat> binar_image = threshold_function(normalized_image); //metodo con funzione threshold (con media )


	// 3) SELEZIONE 2 COMPONENTI DI DIMENSIONE MAGGIORE
	vector<Mat> two_bigger_region = regioni_maggiori(binar_image, normalized_image, 2);


	// 4) EROSIONE 
	vector<Mat> eroded_image = erosione(two_bigger_region);


	// 5) seleziono regione maggiore (cervello)
	vector<Mat> bigger_region = regioni_maggiori(eroded_image, binar_image, 1);

	// 6) DILATO E TAPPO I BUCHI
	vector<Mat> dilated_image = dilatazione(bigger_region, binar_image);
	// Calcolo area totale cervello
	vector<int> area_dilated_image;
	for (int w = 0; w < dilated_image.size(); w++)
	{
		int area_prova = countNonZero(dilated_image[w]);
		area_dilated_image.push_back(area_prova);
	}


	// 7) APPLICO MASCHERA ALLE FOTO ORIGINALI
	vector<Mat> mask_image = maschera(dilated_image, normalized_image);
	statistica(mask_image);

	// 8) NORMALIZZAZIONE/FILTRAGGIO/THRESHOLD
	vector<Mat> filt_image0 = normalizzazione(mask_image);
	vector<Mat> filt_image = filtri(filt_image0);
	vector<Mat> filt_image_bin = binarizzazione_media_no_nero(filt_image0);  //metti immagine non filtrata altrimenti fa il controrno bianco e si unisce alla lesione
	
	
	


	////////// INDIVIDUAZIONE SEED vari metodi/////////////
	int a;
	cout << "Seleziona Metodo per Selezione Seed" << endl;
	cout << "1) Selezione tramite Trasformata Di Hough Circolare" << endl;
	cout << "2) Selezione a mano" << endl;
	cout << "3) Decompressione file testo con coordinate già memorizzate" << endl;

	cin >> a;
	bool valido = true;
	while (valido)
	{
		if (a > 0 && a < 4)
		{
			valido = false;
		}
		else
		{
			cout << "Inserisci valore valido: " << endl;
			cin >> a;
		}
	}


	if (a == 1)
	{
		// 9.1 TRASFORMATA DI HOUGH CIRCOLARE
		trasformata_hough_c(filt_image);   //devo mettere immagine normalizzata e filtrata altrimenti non trova cerchi
	}
	else if (a == 2)
	{
		// 9.2 SALVATAGGIO COORDINATE SEED
		seleziona_seed(filt_image);

		// Compressione coordinate in un file testuale
		//string filename = "salvo_coordinate.txt"; // Regioni scure circondate da chiare
		//saveSeedImagesToFile(seed_images, filename);
		
	}
	else if (a == 3)
	{
		//9.3 CARICA SEED -  DECOMPRESSIONE 
		string filename = "bmp.txt";
		loadSeedImagesFromFile(seed_images, filename);
	}


	//CONVERTE LE IMMAGINI DA OpenCV a ITK
	opencv2itk(filt_image); 
	

	//VISUALIZZA IMMAGINI ITK
	for (size_t i = 0; i < itkImages.size(); ++i) {
		// Converti l'immagine ITK in un oggetto VTK
		using ITKToVTKFilterType = itk::ImageToVTKImageFilter<ImageType>;
		ITKToVTKFilterType::Pointer itkToVtkFilter = ITKToVTKFilterType::New();
		itkToVtkFilter->SetInput(itkImages[i]);
		itkToVtkFilter->Update();

		//// Crea un visualizzatore VTK per l'immagine
		//vtkSmartPointer<vtkImageViewer2> viewer = vtkSmartPointer<vtkImageViewer2>::New();
		//viewer->SetInputData(itkToVtkFilter->GetOutput());

		//// Crea un'interfaccia per il rendering della finestra
		//vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
		//viewer->SetupInteractor(renderWindowInteractor);
		//viewer->Render();
		//viewer->GetRenderer()->ResetCamera();
		//renderWindowInteractor->Start();

		////PER SELEZIONARE IL SEED LE DEVO VISIALIZZZARE CON OPENCV (PARTE DI SOPRA)

	}


	///////////////////SEGMENTAZIONE VARI METODI/////////////////
	int b;
	cout << "Seleziona Metodo di Segmentazione" << endl;
	cout << "1) Region Growing one seed opencv" << endl;
	cout << "2) Algoritmo Watershed (sono necessari 2 seed) " << endl;
	cout << "3) Algoritmo RegionGrowing con ITK - isolated connected (sono necessari 2 seed)" << endl;
	cout << "4) Algoritmo RegionGrowing con ITK - confidence connected" << endl;
	cout << "5) Algoritmo basato sulla trasformata delle distanze" << endl;
	cin >> b;
	valido = true;
	while (valido)
	{
		if (b > 0 && b < 6)
		{
			valido = false;
		}
		else
		{
			cout << "Inserisci valore valido: " << endl;
			cin >> b;
		}
	}
	
	vector<Mat> mask_lesion;
	if (b == 1)
	{
		////10.1 region growing one seed opencv
		for (int w = 0; w < filt_image.size(); w++)
		{
			cv::Point selectedPoint = seed_images[w][0];
			Mat  reg = RegionGrowing(filt_image[w], selectedPoint, (double)50 / 255);
			imshow("Regione Segmentata Algoritmo", reg); 
			waitKey(0); 
			mask_lesion.push_back(reg);
		}
	}
	else if (b == 2)
	{
		//10.2 watershed
		for (int w = 0; w < filt_image0.size(); w++)
		{
			Mat  regW = watershedOpencv(filt_image0[w], normalized_image[w], w);
			/*imshow("Regione Segmentata Algoritmo", regW);
			waitKey(0);*/
			mask_lesion.push_back(regW);
		}
		statistica(mask_lesion);
	}
	else if (b == 3 && (a == 2 || a == 3))
	{
		// 10.3 ALGORITMO RegionGrowing con ITK - isolated connected
		for (int w = 0; w < itkImages.size(); w++)
		{
			
			ImageType::Pointer segmented_image = RegionGrowing_isolated_connected(itkImagesF[w],w); 
			//ImageType::Pointer segmented_image =RegionGrowing_confidence_connected(itkImages[w],w);      
			cv::Mat cvsegmented_image = itk::OpenCVImageBridge::ITKImageToCVMat<ImageType>(segmented_image);  
			mask_lesion.push_back(cvsegmented_image);

		    ////// Visualizza l'immagine utilizzando OpenCV
			imshow("segmented image", cvsegmented_image); 
			waitKey(0);
		
		}

	}
	else if (b == 4 && (a == 2 || a == 3))
	{
		// 10.4 ALGORITMO RegionGrowing con ITK - confidence connected
		for (int w = 0; w < itkImages.size(); w++)
		{
			
			ImageType::Pointer segmented_image =RegionGrowing_confidence_connected(itkImages[w],w);       
			cv::Mat cvsegmented_image = itk::OpenCVImageBridge::ITKImageToCVMat<ImageType>(segmented_image);
			mask_lesion.push_back(cvsegmented_image);
			 //Visualizza l'immagine utilizzando OpenCV
			imshow("segmented image", cvsegmented_image);
			waitKey(0);
		}
	}
	else if (b == 5)
	{
		//10.5 SEGMENTAZIONE CON SEED IN BASE ALLA LUMIOSITA' 
		vector<Mat> mask_lesion1;
		for (int w = 0; w < filt_image.size(); w++)
		{
			Mat  regW = segmentazione_aut(filt_image_bin[w], filt_image[w], normalized_image[w]);
			mask_lesion1.push_back(regW);
		}
		mask_lesion = regioni_maggiori(mask_lesion1, filt_image, 1);
		for (int w = 0; w < mask_lesion.size(); w++)
		{
			imshow("Regione Segmentata Algoritmo", mask_lesion[w]);
			waitKey(0);

		}

	}
	else
	{
		cout << "Seleziona metodo 2) o 3) per la scelta dei seed" << endl;
		return -1;
	}


	////////////////PROCESSING LESIONE///////////////////

	// 12) EVIDENZIO LA LESIONE
	vector<vector<vector<Point>>> contorni;
	contorni = post_processing(mask_lesion, filt_image);

	//// 13) DESCRITTORI GEOMETRICI
	descrittori_geometrici(mask_lesion, area_dilated_image);

	//// 14) DESCRITTORI MORFOLOGICI + calcolo asse maggiore, asse minore, eccentricità
	customMoments(mask_lesion);
	axisInfo(mask_lesion);


	return 0;

}



