#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<filesystem>
#include<string>
#include<thread>
#include <fstream>
#include<chrono>

#include<eigen3/Eigen/Dense>

#define percentage_threshold 0.1

using namespace cv;
using namespace std;
namespace fs=std::filesystem;
using namespace Eigen;

void vocimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process, bool aug);
void cocoimg2contrastive(vector<fs::path> GrayscaleMasks, fs::path coco_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process);
void adeimg2contrastive(vector<fs::path> RawImages, fs::path ade_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process);
void cityimg2contrastive(vector<fs::path> RawImages, fs::path output_dir, fs::path binmask_output_dir, bool print_process);

int main(int argc, char** argv){
    const unsigned int numThreads = std::thread::hardware_concurrency();
    cout << "The system has " << numThreads<<" threads available." << endl;
    cout << "OpenCV version\t: " << CV_VERSION << endl;
    // std::format is temporarily not supported by gcc.
    // Please check `Text formatting` entry under `C++20 library features` table: https://en.cppreference.com/w/cpp/20
    #ifdef _MSC_VER // only msvc supports MSMF
        cout << "Eigen version\t: " << std::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION) << endl;
    #endif
    cout<<"This program is designed to generate binary mask for each object in images from VOC2012, ADE20K, Cityscapes and COCO dataset."<<endl;
    cout<<"It accepts multiple arguments: ./dataset_conv --voc_path [path/to/VOCdevkit/VOC2012] --coco_path [/path/to/coco] --ade_path [/path/to/ADE20K_2021_17_01] --city_path [/path/to/cityscapes contains `/gtFine` and `/leftImg8bit`] --output_dir [desired output directory (default to current dir)], where VOC_root_path is expected to point to VOC2012 folder. Add --save_binmask if you want to save binary masks."<<endl;
    cout<<"Default values of output_path is current path."<<endl;

    auto VOCRootPath=fs::current_path();
    auto ADERootPath=fs::current_path();
    auto CityRootPath=fs::current_path();
    auto COCORootPath=fs::current_path();
    auto GlobalOutputPath=fs::current_path();
    bool write_binmask=false;
    bool flag_voc=false,aug_voc=false,flag_ade=false,flag_coco=false,flag_city=false;
    // If there is input argument.
    if (argc!=1){
        for (size_t i = 1; i < argc; )
        {
            if(string("--voc_path").compare(argv[i])==0){
                flag_voc=true;
                VOCRootPath=argv[i+1];
                cout<<"Given VOCRootPath: "<<VOCRootPath<<endl;
                i=i+2;
                continue;
            }
            else if(string("--aug").compare(argv[i])==0){
                aug_voc=true;
                cout<<"Use `SegmentationClassAug` for VOC."<<endl;
                i=i+1;
                continue;
            }
            else if(string("--ade_path").compare(argv[i])==0){
                flag_ade=true;
                ADERootPath=argv[i+1];
                cout<<"Given ADERootPath: "<<ADERootPath<<endl;
                i=i+2;
                continue;
            }
            else if(string("--coco_path").compare(argv[i])==0){
                flag_coco=true;
                COCORootPath=argv[i+1];
                cout<<"Given COCORootPath: "<<COCORootPath<<endl;
                i=i+2;
                continue;
            }
            else if(string("--city_path").compare(argv[i])==0){
                flag_city=true;
                CityRootPath=argv[i+1];
                cout<<"Given CityscapesRootPath: "<<CityRootPath<<endl;
                i=i+2;
                continue;
            }
            else if(string("--output_dir").compare(argv[i])==0){
                GlobalOutputPath=argv[i+1];
                cout<<"Given OutputPath: "<<GlobalOutputPath<<endl;
                i=i+2;
                continue;
            }
            else if(string("--save_binmask").compare(argv[i])==0){
                write_binmask=true;
                i=i+1;
            }
            else{
                cout<<"Unknown option: "<<argv[i]<<endl;
                return -1;
            }
        }               
    }

    const fs::path OutputSurfix="ContrastivePairs";
    const fs::path OutputSurfix_binmask="ContrastivePairs_binmask";
    if(flag_voc){
        //search for VOC2012 folder in the given VOCRootPath
        fs::path voc_paths;
        if(VOCRootPath.string().find("VOC2012")==string::npos){
            for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(VOCRootPath))
            {
                if(dir_entry.path().string().find("VOC2012")!=string::npos){
                    cout<<"Found '/VOC2012' folder at "<<dir_entry<<endl;
                    voc_paths=dir_entry;
                    VOCRootPath=voc_paths;
                    break;
                }
            }
            if(voc_paths.string().empty()){
                cout<<"Cannot find VOC2012 folder. Please make sure the input VOC_root_path does contain the 'VOC2012' folder."<<endl;
                return -1;
            }
        }

        auto VOC_OutputPath=GlobalOutputPath/OutputSurfix/"voc";
        auto VOC_OutputPath_binmask=GlobalOutputPath/OutputSurfix_binmask/"voc";
        cout<<"Attempt to use VOC dataset path: "<<VOCRootPath<<endl;
        if (write_binmask) {
            fs::create_directories(VOC_OutputPath_binmask);
            cout<<"Binary masks will be saved to: "<<VOC_OutputPath_binmask<<endl;
        }
        else{
            VOC_OutputPath_binmask= fs::path(VOC_OutputPath_binmask.string().erase());
        }

        // interrupt 
        cout<<"Press Enter to start processing VOC2012 dataset or Ctrl-C to exit."<<endl;
        cin.ignore();

        // make sure VOC_OutputPath exists
        fs::create_directories(VOC_OutputPath);
        cout<<"Output path: "<<VOC_OutputPath<<endl;

        fs::path voc_original_mask_path;
        vector<string> train_set_filename;
        if(aug_voc){
            voc_original_mask_path=VOCRootPath/"SegmentationClassAug";
            assert(fs::exists(voc_original_mask_path) && voc_original_mask_path+"dose not exist.");
            // for `SegmentationClassAug`, acquire mask list directly from filenames
            for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(voc_original_mask_path)){
                if(dir_entry.path().string().find(".png")!=string::npos){
                    train_set_filename.push_back(dir_entry.path().stem().string());
                }
            }
        }
        else{
            voc_original_mask_path=VOCRootPath/"SegmentationClass";
            // read image list file
            fs::path train_set_txt=VOCRootPath/"ImageSets"/"Segmentation"/"train.txt";
            assert(fs::exists(train_set_txt) && train_set_txt+"does not exist.");
            ifstream txt;
            txt.open(train_set_txt);
            assert(txt.is_open());
            string tmp_txt;
            while(getline(txt,tmp_txt)){
                train_set_filename.push_back(tmp_txt);
            }
            cout<<train_set_filename.size()<<" training samples retrieved."<<endl;
        }

        // split all images to threads
        vector<fs::path> voc_original_masks;
        vector<vector<fs::path>> split_masks(numThreads);
        size_t t=0;
        for (auto const& onefilename : train_set_filename) 
        {
            if (t>numThreads-1) t=0;
            auto mask_path=voc_original_mask_path/(onefilename+".png");
            if(fs::exists(mask_path)){
                voc_original_masks.push_back(mask_path);
                split_masks[t].push_back(mask_path);
            }
            t++;
        }
        cout<<"In total "<<voc_original_masks.size()<<" original masks."<<endl;
        cout<<"Split for "<<split_masks.size()<<" threads. "<<endl;
        for (size_t i = 0; i < split_masks.size(); i++)
        {
            cout<<"[VOC2012] Thread "<<i<<": "<<split_masks[i].size()<<endl;
        }
        cout<<"files."<<endl;

        // multithread activation
        thread *workers=new thread[numThreads-1];
        for (size_t i = 0; i < numThreads-1; i++)
        {
            workers[i]=thread(vocimg2contrastive,split_masks[i+1],VOCRootPath,VOC_OutputPath,VOC_OutputPath_binmask,false,aug_voc);
        }   
        vocimg2contrastive(split_masks[0],VOCRootPath,VOC_OutputPath,VOC_OutputPath_binmask,true,aug_voc);
        for (auto &one_thread : ranges::subrange(workers,workers+numThreads-1)) one_thread.join();
        delete [] workers;

        // write a filename list of all images
        cout<<"Writing to `VOC_ImgList.csv`. This may take a while."<<endl;
        vector<string> vec_AnchorFilename,vec_NanchorFilename;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(VOC_OutputPath))
        {
            auto one_filename=dir_entry.path().filename();
            if (one_filename.string().find(".jpg")==string::npos) continue;
            if(one_filename.string().find("Nanchor")!=string::npos){
                vec_NanchorFilename.push_back(one_filename.string());
            }
            else{
                vec_AnchorFilename.push_back(one_filename.string());
            }
        }
        // sort filenames
        sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
        sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
        // write to .csv file
        ofstream ImgList;
        ImgList.open(GlobalOutputPath/OutputSurfix/"VOC_ImgList.csv");
        // header
        ImgList<<"anchor,nanchor\n";
        for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
        {
            ImgList<<"voc/"+vec_AnchorFilename[i]<<","<<"voc/"+vec_NanchorFilename[i]<<"\n";
        }
        ImgList.close();    
    }
    if(flag_coco){
        //search for /train2017 folder in the given COCORootPath
        fs::path coco_train_paths;
        if(COCORootPath.string().find("train2017")==string::npos){
            if(fs::exists(COCORootPath/"train2017")){
                cout<<"Found 'train2017' folder at "<<COCORootPath/"train2017"<<endl;
                coco_train_paths=COCORootPath/"train2017";
            }
            if(coco_train_paths.string().empty()){
                cout<<"Cannot find `train2017` folder. Please make sure the input COCO_root_path does contain the `train2017` folder."<<endl;
                return -1;
            }
        }
        auto COCO_OutputPath=GlobalOutputPath/OutputSurfix/"coco";
        auto COCO_OutputPath_binmask=GlobalOutputPath/OutputSurfix_binmask/"coco";
        cout<<"Attempt to use COCO dataset path: "<<COCORootPath<<endl;
        if (write_binmask) {
            fs::create_directories(COCO_OutputPath_binmask);
            cout<<"Binary masks will be saved to: "<<COCO_OutputPath_binmask<<endl;
        }
        else{
            COCO_OutputPath_binmask= fs::path(COCO_OutputPath_binmask.string().erase());
        }

        // interrupt 
        cout<<"Press Enter to start processing COCO dataset or Ctrl-C to exit."<<endl;
        cin.ignore();

        // make sure COCO_OutputPath exists
        fs::create_directories(COCO_OutputPath);
        cout<<"Output path: "<<COCO_OutputPath<<endl;

        // create a list of mask paths
        vector<fs::path> gray_mask_paths;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(COCORootPath/"stuffthingmaps_trainval2017"/"train2017"))
        {
            if(dir_entry.path().string().find(".png")!=string::npos){
                gray_mask_paths.push_back(dir_entry);
                // cout<<dir_entry<<endl;
            }
        }
        cout<<"In total "<<gray_mask_paths.size()<<" original masks."<<endl;
        
        // split all images to threads
        vector<vector<fs::path>> split_masks(numThreads);
        size_t num_samples_thread=gray_mask_paths.size()/numThreads;
        for (size_t i=0; i<numThreads-1;i++) 
        {
            vector<fs::path> one_thread_samples(gray_mask_paths.begin()+num_samples_thread*i,gray_mask_paths.begin()+num_samples_thread*(i+1));
            split_masks[i]=one_thread_samples;
        }
        vector<fs::path> main_thread_samples(gray_mask_paths.begin()+num_samples_thread*(numThreads-1),gray_mask_paths.end());
        split_masks[numThreads-1]=main_thread_samples;
        cout<<"Split for "<<split_masks.size()<<" threads. "<<endl;
        for (size_t i = 0; i < split_masks.size(); i++)
        {
            cout<<"[COCO] Thread "<<i<<": "<<split_masks[i].size()<<endl;
        }
        cout<<"files."<<endl;

        // multithread activation
        thread *workers=new thread[numThreads-1];
        for (size_t i = 0; i < numThreads-1; i++)
        {
            workers[i]=thread(cocoimg2contrastive,split_masks[i],COCORootPath,COCO_OutputPath,COCO_OutputPath_binmask,false);
        }   
        cocoimg2contrastive(split_masks[numThreads-1],COCORootPath,COCO_OutputPath,COCO_OutputPath_binmask,true);
        for (auto &one_thread : ranges::subrange(workers,workers+numThreads-1)) one_thread.join();
        delete [] workers;

        // write a filename list of all images
        vector<string> vec_AnchorFilename,vec_NanchorFilename;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(COCO_OutputPath))
            {
                auto one_filename=dir_entry.path().filename();
                if (one_filename.string().find(".jpg")==string::npos) continue;
                if(one_filename.string().find("Nanchor")!=string::npos){
                    vec_NanchorFilename.push_back(one_filename.string());
                }
                else{
                    vec_AnchorFilename.push_back(one_filename.string());
                }
            }
        // sort filenames
        sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
        sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
        // write to .csv file
        ofstream ImgList;
        ImgList.open(GlobalOutputPath/OutputSurfix/"COCO_ImgList.csv");
        // header
        ImgList<<"anchor,nanchor\n";
        for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
        {
            ImgList<<"coco/"+vec_AnchorFilename[i]<<","<<"coco/"+vec_NanchorFilename[i]<<"\n";
        }
        ImgList.close();    
    }
    if(flag_ade){
        //search for /images/ADE/training folder in the given ADERootPath
        fs::path ade_train_paths;
        fs::path images_ade_training=fs::path("images")/"ADE"/"training";
        if(ADERootPath.string().find(images_ade_training.string())==string::npos){
            for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(ADERootPath))
            {
                if(dir_entry.path().string().find(images_ade_training.string())!=string::npos){
                    cout<<"Found '/images/ADE/training' folder at "<<dir_entry<<endl;
                    ade_train_paths=dir_entry;
                    break;
                }
            }
            if(ade_train_paths.string().empty()){
                cout<<"Cannot find /images/ADE/training folder. Please make sure the input ADE_root_path does contain the '/images/ADE/training' folder."<<endl;
                return -1;
            }
        }

        auto ADE_OutputPath=GlobalOutputPath/OutputSurfix/"ade20k";
        auto ADE_OutputPath_binmask=GlobalOutputPath/OutputSurfix_binmask/"ade20k";
        cout<<"Attempt to use ADE dataset path: "<<ade_train_paths<<endl;
        if (write_binmask) {
            fs::create_directories(ADE_OutputPath_binmask);
            cout<<"Binary masks will be saved to: "<<ADE_OutputPath_binmask<<endl;
        }
        else{
            ADE_OutputPath_binmask= fs::path(ADE_OutputPath_binmask.string().erase());
        }

        // interrupt 
        cout<<"Press Enter to start processing ADE20K dataset or Ctrl-C to exit."<<endl;
        cin.ignore();

        // make sure ADE_OutputPath exists
        fs::create_directories(ADE_OutputPath);
        cout<<"Output path: "<<ADE_OutputPath<<endl;

        // create a list of raw image paths
        vector<fs::path> raw_image_paths;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(ade_train_paths))
        {
            if(dir_entry.path().string().find(".jpg")!=string::npos){
                raw_image_paths.push_back(dir_entry);
            }
        }
        cout<<"In total "<<raw_image_paths.size()<<" raw images."<<endl;

        // split all images to threads
        vector<vector<fs::path>> split_masks(numThreads);
        size_t t=0;
        size_t num_samples_thread=raw_image_paths.size()/numThreads;
        for (size_t i=0; i<numThreads-1;i++) 
        {
            vector<fs::path> one_thread_samples(raw_image_paths.begin()+num_samples_thread*i,raw_image_paths.begin()+num_samples_thread*(i+1));
            split_masks[i]=one_thread_samples;
        }
        vector<fs::path> main_thread_samples(raw_image_paths.begin()+num_samples_thread*(numThreads-1),raw_image_paths.end());
        split_masks[numThreads-1]=main_thread_samples;
        cout<<"Split for "<<split_masks.size()<<" threads. "<<endl;
        for (size_t i = 0; i < split_masks.size(); i++)
        {
            cout<<"[ADE20K] Thread "<<i<<": "<<split_masks[i].size()<<endl;
        }
        cout<<"files."<<endl;

        // multithread activation
        thread *workers=new thread[numThreads-1];
        for (size_t i = 0; i < numThreads-1; i++)
        {
            workers[i]=thread(adeimg2contrastive,split_masks[i+1],ade_train_paths,ADE_OutputPath,ADE_OutputPath_binmask,false);
        }   
        adeimg2contrastive(split_masks[0],ade_train_paths,ADE_OutputPath,ADE_OutputPath_binmask,true);
        for (auto &one_thread : ranges::subrange(workers,workers+numThreads-1)) one_thread.join();
        delete [] workers;

        // write a filename list of all images
        cout<<"Writing to `ADE_ImgList.csv`. This may take a while."<<endl;
        vector<string> vec_AnchorFilename,vec_NanchorFilename;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(ADE_OutputPath))
            {
                auto one_filename=dir_entry.path().filename();
                if (one_filename.string().find(".jpg")==string::npos) continue;
                if(one_filename.string().find("Nanchor")!=string::npos){
                    vec_NanchorFilename.push_back(one_filename.string());
                }
                else{
                    vec_AnchorFilename.push_back(one_filename.string());
                }
            }
        // sort filenames
        sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
        sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
        // write to .csv file
        ofstream ImgList;
        ImgList.open(GlobalOutputPath/OutputSurfix/"ADE_ImgList.csv");
        // header
        ImgList<<"anchor,nanchor\n";
        for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
        {
            ImgList<<"ade20k/"+vec_AnchorFilename[i]<<","<<"ade20k/"+vec_NanchorFilename[i]<<"\n";
        }
        ImgList.close();    
    }
    if(flag_city){
        //search for /gtFine and /leftImg8bit folders under the given CityRootPath
        fs::path city_train_paths=CityRootPath;
        fs::path city_label_paths;
        fs::path city_img_paths;
        for (auto const& one_subdir : fs::directory_iterator(city_train_paths))
        {
            if (fs::is_directory(one_subdir)){
                if (one_subdir.path().string().find("gtFine")!=string::npos){
                    city_label_paths=one_subdir.path()/"train";
                    if (!fs::exists(city_label_paths)) {
                        cout<<"Cannot find Cityscapes `/gtFine/train` for labels under "<<city_label_paths<<". Please check.";
                        return -1;
                    }
                }
                else if (one_subdir.path().string().find("leftImg8bit")!=string::npos){
                    city_img_paths=one_subdir.path()/"train";
                    if (!fs::exists(city_img_paths)) {
                        cout<<"Cannot find Cityscapes `/leftImg8bit/train` for raw images under "<<city_img_paths<<". Please check.";
                        return -1;
                    }
                }
            }
        }
        

        auto city_OutputPath=GlobalOutputPath/OutputSurfix/"cityscapes";
        auto city_OutputPath_binmask=GlobalOutputPath/OutputSurfix_binmask/"cityscapes";
        cout<< "Attempt to use Cityscapes dataset path: " << city_train_paths <<endl;
        if (write_binmask) {
            fs::create_directories(city_OutputPath_binmask);
            cout<<"Binary masks will be saved to: "<<city_OutputPath_binmask<<endl;
        }
        else{
            city_OutputPath_binmask= fs::path(city_OutputPath_binmask.string().erase());
        }

        // interrupt 
        cout<<"Press Enter to start processing Cityscapes dataset or Ctrl-C to exit."<<endl;
        cin.ignore();

        // make sure city_OutputPath exists
        fs::create_directories(city_OutputPath);
        cout<<"Output path: "<<city_OutputPath<<endl;

        // create a list of raw image paths
        vector<fs::path> raw_image_paths;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(city_img_paths))
        {
            if(dir_entry.path().string().find("_leftImg8bit.png")!=string::npos){
                raw_image_paths.push_back(dir_entry);
            }
        }

        // split all images to threads
        vector<vector<fs::path>> split_imgs(numThreads);
        size_t t=0;
        for (auto const& OneRawImagePath : raw_image_paths) 
        {
            if (t>numThreads-1) t=0;
            if(fs::exists(OneRawImagePath)){
                split_imgs[t].push_back(OneRawImagePath);
            }
            t++;
        }
        cout<<"In total "<<raw_image_paths.size()<<" raw images."<<endl;
        cout<<"Split for "<<split_imgs.size()<<" threads. "<<endl;
        for (size_t i = 0; i < split_imgs.size(); i++)
        {
            cout<<"[Cityscapes] Thread "<<i<<": "<<split_imgs[i].size()<<endl;
        }
        cout<<"files."<<endl;

        // multithread activation
        thread *workers=new thread[numThreads-1];
        for (size_t i = 0; i < numThreads-1; i++)
        {
            workers[i]=thread(cityimg2contrastive,split_imgs[i+1],city_OutputPath,city_OutputPath_binmask,false);
        }   
        cityimg2contrastive(split_imgs[0],city_OutputPath,city_OutputPath_binmask,true);
        for (auto &one_thread : ranges::subrange(workers,workers+numThreads-1)) one_thread.join();
        delete [] workers;

        // write a filename list of all images
        cout<<"Writing to `Cityscapes_ImgList.csv`. This may take a while."<<endl;
        vector<string> vec_AnchorFilename,vec_NanchorFilename;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(city_OutputPath))
            {
                auto one_filename=dir_entry.path().filename();
                if (one_filename.string().find(".png")==string::npos) continue;
                if(one_filename.string().find("Nanchor")!=string::npos){
                    vec_NanchorFilename.push_back(one_filename.string());
                }
                else{
                    vec_AnchorFilename.push_back(one_filename.string());
                }
            }
        // sort filenames
        sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
        sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
        // write to .csv file
        ofstream ImgList;
        ImgList.open(GlobalOutputPath/OutputSurfix/"Cityscapes_ImgList.csv");
        // header
        ImgList<<"anchor,nanchor\n";
        for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
        {
            ImgList<<"cityscapes/"+vec_AnchorFilename[i]<<","<<"cityscapes/"+vec_NanchorFilename[i]<<"\n";
        }
        ImgList.close();    
    }
    return 0;
}

void vocimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process, bool aug){
    // Following voc_colormap is from
    // https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
    // black background is removed
    vector<vector<uint>> voc_colormap;
    if (aug){
        // From the README of Semantic Boundaries Dataset(SBD): Pixels that belong to category k have value k, pixels that do not belong to any category have value 0.
        for (uint i = 0; i < 20; i++) 
        {
            voc_colormap.push_back({i+1,i+1,i+1});
        }
    }
    else{
        voc_colormap={
            {128, 0, 0},
            {0, 128, 0},
            {128, 128, 0},
            {0, 0, 128},
            {128, 0, 128},
            {0, 128, 128},
            {128, 128, 128},
            {64, 0, 0},
            {192, 0, 0},
            {64, 128, 0},
            {192, 128, 0},
            {64, 0, 128},
            {192, 0, 128},
            {64, 128, 128},
            {192, 128, 128},
            {0, 64, 0},
            {128, 64, 0},
            {0, 192, 0},
            {128, 192, 0},
            {0, 64, 128}
        };
    }
    
    auto jpegPath=voc_root/"JPEGImages";
    size_t counter=0;
    for(auto const& OneColorfulMask: ColorfulMasks){
        auto corres_jpeg=jpegPath/(OneColorfulMask.stem().string()+".jpg");

        if (!fs::exists(corres_jpeg)){
            cout<<"File "<<corres_jpeg<<" does not exist."<<endl;
            abort();
        }
        Mat jpeg=imread(corres_jpeg.string());
        Mat tmp_mask=imread(OneColorfulMask.string());
        cvtColor(tmp_mask,tmp_mask,COLOR_BGR2RGB);

        vector<vector<Vector2i>> tmp_pixel_class(voc_colormap.size());// we would ignore black background and object edge
        // cout<<"Start pixel-wise match."<<endl;
        for (size_t r = 0; r < tmp_mask.rows; r++)
        {
            for (size_t c = 0; c < tmp_mask.cols; c++)
            {
                auto pixelRGB=tmp_mask.at<Vec3b>(r,c);
                vector<uint> tmp_RGB={pixelRGB[0],pixelRGB[1],pixelRGB[2]};
                Vector2i coordinates(r,c);
                for (size_t i = 0; i < voc_colormap.size(); i++)
                {
                    if(voc_colormap[i]==tmp_RGB){
                        tmp_pixel_class[i].push_back(coordinates);
                        // cout<<"("<<r<<","<<c<<")"<<endl;
                    }
                }                
            }            
        }
        // cout<<"Pixel-wise match finished."<<endl;
        
        // generate binary mask
        // cout<<"Generate binary masks."<<endl;
        vector<Mat> bin_masks;
        for (size_t i = 0; i < tmp_pixel_class.size(); i++)
        {
            if(tmp_pixel_class[i].empty()){
                continue;
            }
            else if(tmp_pixel_class[i].size()<=percentage_threshold*tmp_mask.rows*tmp_mask.cols){
                // discard current class if it occupies less than 20% of raw image content
                continue;
            }
            else{
                // binary mask, initialized to pure black
                Mat tmp_bin_mask(tmp_mask.size(),CV_8UC3,Scalar(0,0,0));
                for (size_t j = 0; j < tmp_pixel_class[i].size(); j++)
                {
                    size_t pixel_row=tmp_pixel_class[i][j](0);
                    size_t pixel_col=tmp_pixel_class[i][j](1);
                    // cout<<"Point "<<j<<": ("<<pixel_row<<","<<pixel_col<<")"<<endl;
                    tmp_bin_mask.at<Vec3b>(pixel_row,pixel_col)={255,255,255};
                }
                // save binary mask if needed
                if (!binmask_output_dir.empty()){
                    auto bin_mask_filename=binmask_output_dir/(OneColorfulMask.stem().string()+"_binmask"+to_string(i)+".jpg");
                    auto nbin_mask_filename=binmask_output_dir/(OneColorfulMask.stem().string()+"_nbinmask"+to_string(i)+".jpg");
                    imwrite(bin_mask_filename.string(),tmp_bin_mask);
                    imwrite(nbin_mask_filename.string(),~tmp_bin_mask);
                }
                
                bin_masks.push_back(tmp_bin_mask);
            }
        }
        // cout<<"Generate binary masks finished."<<endl;        
        
        vector<Mat> anchor,Nanchor;
        for (size_t i = 0; i < bin_masks.size(); i++)
        {
            auto invert_bin_mask=~bin_masks[i];
            Mat tmp_anchor,tmp_Nanchor;
            auto anchor_filename=output_dir/(OneColorfulMask.stem().string()+"_anchor"+to_string(i)+".jpg");
            auto Nanchor_filename=output_dir/(OneColorfulMask.stem().string()+"_Nanchor"+to_string(i)+".jpg");
            // cout<<"before bitwise_and."<<endl;
            bitwise_and(jpeg,bin_masks[i],tmp_anchor);
            // cout<<"between bitwise_and."<<endl;
            bitwise_and(jpeg,invert_bin_mask,tmp_Nanchor);
            // cout<<"after bitwise_and."<<endl;
            imwrite(anchor_filename.string(),tmp_anchor);
            imwrite(Nanchor_filename.string(),tmp_Nanchor);
        }
        counter++;
        if (print_process){
            double process=counter/(double)ColorfulMasks.size()*100;
            cout<<"[VOC2012] "<<process<<"%"<<endl;
        }
    }
}

void cocoimg2contrastive(vector<fs::path> GrayscaleMasks, fs::path coco_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process){
    // 8-bit gray sacale mask of coco provided in stuffthingmaps_trainval2017.zip
    // by https://github.com/nightrome/cocostuff#downloads. The authors provided a lable-color map at
    // https://github.com/nightrome/cocostuff/blob/master/labels.txt, but it is quite misleading:
    // "unlabeled" should be 255 in grayscale annotations, the reset should be
    // 0: person
    // 1: bicycle
    // ...
    // 181: wood
    // generate a 0 to 181 in an int container (255 stands for "no-label")
    vector<int> coco_colormap(182);
    for (size_t i = 0; i < 182; i++)
    {
        coco_colormap[i]=i;
    }
    auto RawImagePath=coco_root/"train2017";
    size_t counter=0;
    for (auto const& OneGrayMask:GrayscaleMasks)
    {
        auto corres_jpg=RawImagePath/(OneGrayMask.stem().string()+".jpg");
        if (!fs::exists(corres_jpg)){
            cout<<"File "<<corres_jpg<<" does not exist."<<endl;
            abort();
        }
        Mat jpeg=imread(corres_jpg.string(),IMREAD_COLOR);
        Mat tmp_mask=imread(OneGrayMask.string(),IMREAD_GRAYSCALE );

        vector<vector<Vector2i>> tmp_pixel_class(coco_colormap.size());// we would ignore unlabeled pixel (255)
        // cout<<"Start pixel-wise match."<<endl;
        for (size_t r = 0; r < tmp_mask.rows; r++)
        {
            for (size_t c = 0; c < tmp_mask.cols; c++)
            {
                int GrayValue=tmp_mask.at<uchar>(r,c);
                Vector2i coordinates(r,c);
                for (size_t i = 0; i < coco_colormap.size(); i++)
                {
                    if(coco_colormap[i]==GrayValue){
                        tmp_pixel_class[i].push_back(coordinates);
                        // cout<<"("<<r<<","<<c<<")"<<endl;
                    }
                }                
            }            
        }
        // cout<<"Pixel-wise match finished."<<endl;

        // generate binary mask
        // cout<<"Generate binary masks."<<endl;
        vector<Mat> bin_masks;
        for (size_t i = 0; i < tmp_pixel_class.size(); i++)
        {
            if(tmp_pixel_class[i].empty()){
                continue;
            }
            else if(tmp_pixel_class[i].size()<=percentage_threshold*tmp_mask.rows*tmp_mask.cols){
                // discard current class if it occupies less than 20% of raw image content
                continue;
            }
            else{
                // binary mask, initialized to pure black
                Mat tmp_bin_mask(tmp_mask.size(),CV_8UC3,Scalar(0,0,0));
                for (size_t j = 0; j < tmp_pixel_class[i].size(); j++)
                {
                    size_t pixel_row=tmp_pixel_class[i][j](0);
                    size_t pixel_col=tmp_pixel_class[i][j](1);
                    // cout<<"Point "<<j<<": ("<<pixel_row<<","<<pixel_col<<")"<<endl;
                    tmp_bin_mask.at<Vec3b>(pixel_row,pixel_col)={255,255,255};
                }
                // save binary mask if needed
                if (!binmask_output_dir.empty()){
                    auto bin_mask_filename=binmask_output_dir/(OneGrayMask.stem().string()+"_binmask"+to_string(i)+".jpg");
                    auto nbin_mask_filename=binmask_output_dir/(OneGrayMask.stem().string()+"_nbinmask"+to_string(i)+".jpg");
                    imwrite(bin_mask_filename.string(),tmp_bin_mask);
                    imwrite(nbin_mask_filename.string(),~tmp_bin_mask);
                }
                
                bin_masks.push_back(tmp_bin_mask);
            }
        }
        // cout<<"Generate binary masks finished."<<endl;        

        vector<Mat> anchor,Nanchor;
        for (size_t i = 0; i < bin_masks.size(); i++)
        {
            auto invert_bin_mask=~bin_masks[i];
            Mat tmp_anchor,tmp_Nanchor;
            auto anchor_filename=output_dir/(OneGrayMask.stem().string()+"_anchor"+to_string(i)+".jpg");
            auto Nanchor_filename=output_dir/(OneGrayMask.stem().string()+"_Nanchor"+to_string(i)+".jpg");
            // cout<<"before bitwise_and."<<endl;
            bitwise_and(jpeg,bin_masks[i],tmp_anchor);
            // cout<<"between bitwise_and."<<endl;
            bitwise_and(jpeg,invert_bin_mask,tmp_Nanchor);
            // cout<<"after bitwise_and."<<endl;
            imwrite(anchor_filename.string(),tmp_anchor);
            imwrite(Nanchor_filename.string(),tmp_Nanchor);
        }
        counter++;
        if (print_process && counter%100==0){
            double process=counter/(double)GrayscaleMasks.size()*100;
            cout<<"[COCO] "<<process<<"%"<<endl;
        }
    }    
}

void adeimg2contrastive(vector<fs::path> RawImages, fs::path ade_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process){
    // design of this function is referred to ADE20K dataset structure
    // https://github.com/CSAILVision/ADE20K#structure
    for (size_t i = 0; i < RawImages.size(); i++)
    {
        auto OneRawImage=RawImages[i];
        Mat RawImageMat=imread(OneRawImage.string(),IMREAD_COLOR);
        auto SegMaskDir=OneRawImage.parent_path()/OneRawImage.stem();
        if (!fs::exists(SegMaskDir)) cout<<SegMaskDir<<" does not exist."<<endl;
        size_t k=0;
        for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{SegMaskDir}) 
        {
            if(dir_entry.path().string().find(".png")!=string::npos &&
               dir_entry.path().string().find("instance_")!=string::npos ){
                Mat OneSegMask=imread(dir_entry.path().string(),IMREAD_GRAYSCALE);
            
                vector<Vector2i> tmp_pixel_class;// only take pixels with value 255
                // cout<<"Start pixel-wise match."<<endl;
                for (size_t r = 0; r < OneSegMask.rows; r++)
                {
                    for (size_t c = 0; c < OneSegMask.cols; c++)
                    {
                        int GrayValue=OneSegMask.at<uchar>(r,c);
                        Vector2i coordinates(r,c);
                        if(255==GrayValue) tmp_pixel_class.push_back(coordinates);
                        // cout<<"("<<r<<","<<c<<")"<<endl;
                    }
                }

                if(tmp_pixel_class.size()<=percentage_threshold*OneSegMask.rows*OneSegMask.cols) {
                    // cout<<"skip"<<endl;
                    continue;
                }
                else {
                    Mat tmp_bin_mask(OneSegMask.size(),CV_8UC3,Scalar(0,0,0));
                    for (auto const& OneCoordinate:tmp_pixel_class)
                    {
                        tmp_bin_mask.at<Vec3b>(OneCoordinate(0),OneCoordinate(1))={255,255,255};
                    }   
                    // save binary mask if needed
                    if (!binmask_output_dir.empty()){
                        auto bin_mask_filename=binmask_output_dir/(OneRawImage.stem().string()+"_binmask"+to_string(k)+".jpg");
                        auto nbin_mask_filename=binmask_output_dir/(OneRawImage.stem().string()+"_nbinmask"+to_string(k)+".jpg");
                        imwrite(bin_mask_filename.string(),tmp_bin_mask);
                        imwrite(nbin_mask_filename.string(),~tmp_bin_mask);
                    }
                    auto invert_bin_mask=~tmp_bin_mask;
                    Mat tmp_anchor,tmp_Nanchor;
                    auto anchor_filename=output_dir/(OneRawImage.stem().string()+"_anchor"+to_string(k)+".jpg");
                    auto Nanchor_filename=output_dir/(OneRawImage.stem().string()+"_Nanchor"+to_string(k)+".jpg");
                    k++;
                    // cout<<"before bitwise_and."<<endl;
                    bitwise_and(RawImageMat,tmp_bin_mask,tmp_anchor);
                    // cout<<"between bitwise_and."<<endl;
                    bitwise_and(RawImageMat,invert_bin_mask,tmp_Nanchor);
                    // cout<<"after bitwise_and."<<endl;
                    imwrite(anchor_filename.string(),tmp_anchor);
                    imwrite(Nanchor_filename.string(),tmp_Nanchor);                        
                }
                // cout<<"Pixel-wise match finished."<<endl;
            }
        }

        if(print_process && i%100==0) {
            double process=i/(double)RawImages.size()*100;
            cout<<"[ADE20k] "<<process<<"%"<<endl;
        }
    }    
}

void cityimg2contrastive(vector<fs::path> RawImages, fs::path output_dir, fs::path binmask_output_dir, bool print_process){
    // design of this function is referred to Cityscapes dataset structure
    vector<vector<int>> city_colormap={
        // {0,0,0}, //ignore black background
        {111, 74, 0},
        {81, 0, 81},
        {128, 64, 128},
        {244, 35, 232},
        {250, 170, 160},
        {230, 150, 140},
        {70, 70, 70},
        {102, 102, 156},
        {190, 153, 153},
        {180, 165, 180},
        {150, 100, 100},
        {150, 120, 90},
        {153, 153, 153},
        {250, 170, 30},
        {220, 220, 0},
        {107, 142, 35},
        {152, 251, 152},
        {70, 130, 180},
        {220, 20, 60},
        {255, 0, 0},
        {0, 0, 142},
        {0, 0, 70},
        {0, 60, 100},
        {0, 0, 90},
        {0, 0, 110},
        {0, 80, 100},
        {0, 0, 230},
        {119, 11, 32},
        {0, 0, 142}
    };
    size_t suffix_len=string("leftImg8bit.png").length();
    for (size_t i = 0; i < RawImages.size(); i++)
    {
        auto OneRawImage=RawImages[i].string();
        Mat RawImageMat=imread(OneRawImage);
        string OneColorMask=OneRawImage;

        // cout<<"Start getting corresponding mask path."<<endl;
        do{
            OneColorMask=OneColorMask.replace(OneColorMask.find("leftImg8bit"),suffix_len-4,"gtFine");
            // cout<<OneColorMask<<endl;
        }
        while(OneColorMask.find("leftImg8bit")!=string::npos);//replace `leftImg8bit` with `gtFine`
        // cout<<"After getting corresponding mask path."<<endl;
        

        auto SegMaskDir=OneColorMask.insert(OneColorMask.find(".png"),"_color");
        if (!fs::exists(SegMaskDir)) {
            cout<<SegMaskDir<<" does not exist."<<endl;
            abort();
        }
        Mat OneSegMask=imread(SegMaskDir);
        cvtColor(OneSegMask,OneSegMask,COLOR_BGR2RGB);

        vector<vector<Vector2i>> tmp_pixel_class(city_colormap.size());// only take pixels with value 255
        // cout<<"Start pixel-wise match."<<endl;
        for (size_t r = 0; r < OneSegMask.rows; r++)
        {
            for (size_t c = 0; c < OneSegMask.cols; c++)
            {
                auto pixelRGB=OneSegMask.at<Vec3b>(r,c);
                vector<int> tmp_RGB={pixelRGB[0],pixelRGB[1],pixelRGB[2]};
                Vector2i coordinates(r,c);
                for (size_t i = 0; i < city_colormap.size(); i++)
                {
                    if(city_colormap[i]==tmp_RGB){
                        tmp_pixel_class[i].push_back(coordinates);
                        // cout<<"("<<r<<","<<c<<")"<<endl;
                    }
                }
            }
        }

        // generate binary mask
        // cout<<"Generate binary masks."<<endl;
        vector<Mat> bin_masks;
        for (size_t i = 0; i < tmp_pixel_class.size(); i++)
        {
            if(tmp_pixel_class[i].empty()){
                continue;
            }
            else if(tmp_pixel_class[i].size()<=percentage_threshold*OneSegMask.rows*OneSegMask.cols){
                // discard current class if it occupies less than 20% of raw image content
                continue;
            }
            else{
                // binary mask, initialized to pure black
                Mat tmp_bin_mask(OneSegMask.size(),CV_8UC3,Scalar(0,0,0));
                for (size_t j = 0; j < tmp_pixel_class[i].size(); j++)
                {
                    size_t pixel_row=tmp_pixel_class[i][j](0);
                    size_t pixel_col=tmp_pixel_class[i][j](1);
                    // cout<<"Point "<<j<<": ("<<pixel_row<<","<<pixel_col<<")"<<endl;
                    tmp_bin_mask.at<Vec3b>(pixel_row,pixel_col)={255,255,255};
                }
                // save binary mask if needed
                if (!binmask_output_dir.empty()){
                    auto bin_mask_filename=binmask_output_dir/(fs::path(SegMaskDir).stem().string()+"_binmask"+to_string(i)+".jpg");
                    auto nbin_mask_filename=binmask_output_dir/(fs::path(SegMaskDir).stem().string()+"_nbinmask"+to_string(i)+".jpg");
                    imwrite(bin_mask_filename.string(),tmp_bin_mask);
                    imwrite(nbin_mask_filename.string(),~tmp_bin_mask);
                }
                
                bin_masks.push_back(tmp_bin_mask);
            }
        }
        // cout<<"Generate binary masks finished."<<endl;        

        vector<Mat> anchor,Nanchor;
        for (size_t i = 0; i < bin_masks.size(); i++)
        {
            auto invert_bin_mask=~bin_masks[i];
            Mat tmp_anchor,tmp_Nanchor;
            auto anchor_filename=output_dir/(fs::path(SegMaskDir).stem().string()+"_anchor"+to_string(i)+".png");
            auto Nanchor_filename=output_dir/(fs::path(SegMaskDir).stem().string()+"_Nanchor"+to_string(i)+".png");
            // cout<<"before bitwise_and."<<endl;
            bitwise_and(RawImageMat,bin_masks[i],tmp_anchor);
            // cout<<"between bitwise_and."<<endl;
            bitwise_and(RawImageMat,invert_bin_mask,tmp_Nanchor);
            // cout<<"after bitwise_and."<<endl;
            imwrite(anchor_filename.string(),tmp_anchor);
            imwrite(Nanchor_filename.string(),tmp_Nanchor);
        }

        if(print_process) {
            double process=i/(double)RawImages.size()*100;
            cout<<"[Cityscapes] "<<process<<"%"<<endl;
        }
    }    
}