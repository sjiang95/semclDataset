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

using namespace cv;
using namespace std;
namespace fs=std::filesystem;
using namespace Eigen;

void vocimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir);
void cocoimg2contrastive(vector<fs::path> GrayscaleMasks, fs::path coco_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process);
void adeimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process);

int main(int argc, char** argv){
    unsigned int numThreads = std::thread::hardware_concurrency();
    cout << "The system has " << numThreads<<" threads available." << endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout<<"This program is designed to generate binary mask for each object in images from VOC 2012 dataset."<<endl;
    cout<<"You should make a copy of original VOC2012 dataset since this program would output 'binary_mask' to VOC2012 folder."<<endl;
    cout<<"It accepts multiple arguments: voc_conv --voc_path [VOC_root_path] --mode [mode], where VOC_root_path is expected to point to VOC2012 folder. Add --save_binmask if you want to save binary masks."<<endl;
    cout<<"Set mode as either semantic or instance. Default is semantic."<<endl;
    cout<<"Default values of VOC_root_path and output_path are current path."<<endl;

    auto VOCRootPath=fs::current_path();
    auto ADERootPath=fs::current_path();
    auto COCORootPath=fs::current_path();
    auto GlobalOutputPath=fs::current_path();
    bool write_binmask=false;
    bool flag_voc=false,flag_ade=false,flag_coco=false;
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
            else if(string("--output_dir").compare(argv[i])==0){
                flag_coco=true;
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
            if(VOCRootPath.string().find("/VOC2012")==string::npos){
                for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(VOCRootPath))
                {
                    if(dir_entry.path().string().find("VOC2012")!=string::npos){
                        cout<<"Find '/VOC2012' folder at "<<dir_entry<<endl;
                        voc_paths=dir_entry;
                        VOCRootPath=voc_paths;
                        break;
                    }
                }
                if(voc_paths.string().empty()){
                    cout<<"Cannot find VOC2012 folder. Please make sure the input VOC_root_path does contain the '/VOC2012' folder."<<endl;
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
            cout<<"Press Enter to start processing VOC2012 dataset."<<endl;
            while(true){
                if(waitKey(100)==13) break;
            }

            // timer
            auto TP=chrono::high_resolution_clock::now();

            // make sure VOC_OutputPath exists
            fs::create_directories(VOC_OutputPath);
            cout<<"Output path: "<<VOC_OutputPath<<endl;

            // read image list file
            fs::path train_set_txt=VOCRootPath/"ImageSets/Segmentation/train.txt";
            ifstream txt;
            txt.open(train_set_txt);
            vector<string> train_set_filename;
            assert(txt.is_open());
            string tmp_txt;
            while(getline(txt,tmp_txt)){
                train_set_filename.push_back(tmp_txt);
            }

            // split all images to threads
            vector<fs::path> voc_original_masks;
            vector<vector<fs::path>> split_masks(numThreads);
            size_t t=0;
            fs::path voc_original_mask_path=VOCRootPath/"SegmentationClass";
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
            thread workers[numThreads-1];
            for (size_t i = 0; i < numThreads-1; i++)
            {
                workers[i]=thread(vocimg2contrastive,split_masks[i+1],VOCRootPath,VOC_OutputPath,VOC_OutputPath_binmask);
            }   
            vocimg2contrastive(split_masks[0],VOCRootPath,VOC_OutputPath,VOC_OutputPath_binmask);
            for (auto &one_thread : workers) one_thread.join();

            // write a filename list of all images
            vector<string> vec_AnchorFilename,vec_NanchorFilename;
            for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(VOC_OutputPath))
                {
                    auto one_filename=dir_entry.path().filename();
                    if (one_filename.string().find(".jpg")==string::npos) continue;
                    if(one_filename.string().find("Nanchor")!=string::npos){
                        vec_NanchorFilename.push_back(one_filename);
                    }
                    else{
                        vec_AnchorFilename.push_back(one_filename);
                    }
                }
            // sort filenames
            sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
            sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
            // write to .csv file
            ofstream ImgList;
            ImgList.open(VOC_OutputPath/"ImgList.csv");
            // header
            ImgList<<"anchor,nanchor\n";
            for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
            {
                ImgList<<vec_AnchorFilename[i]<<","<<vec_NanchorFilename[i]<<"\n";
            }
            ImgList.close();    
            chrono::duration<long double,std::ratio<1,1>> duration=chrono::high_resolution_clock::now()-TP;
    }
    else if(flag_coco){
        //search for /train2017 folder in the given COCORootPath
        fs::path coco_train_paths;
        if(COCORootPath.string().find("/train2017")==string::npos){
            for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(COCORootPath))
            {
                if(dir_entry.path().string().find("/train2017")!=string::npos){
                    cout<<"Find '/train2017' folder at "<<dir_entry<<endl;
                    coco_train_paths=dir_entry;
                    break;
                }
            }
            if(coco_train_paths.string().empty()){
                cout<<"Cannot find /train2017 folder. Please make sure the input COCO_root_path does contain the '/train2017' folder."<<endl;
                return -1;
            }
        }
        auto COCO_OutputPath=GlobalOutputPath/OutputSurfix/"coco";
        auto COCO_OutputPath_binmask=GlobalOutputPath/OutputSurfix_binmask/"coco";
        cout<<"Attempt to use VOC dataset path: "<<VOCRootPath<<endl;
        if (write_binmask) {
            fs::create_directories(COCO_OutputPath_binmask);
            cout<<"Binary masks will be saved to: "<<COCO_OutputPath_binmask<<endl;
        }
        else{
            COCO_OutputPath_binmask= fs::path(COCO_OutputPath_binmask.string().erase());
        }

        // interrupt 
        cout<<"Press Enter to start processing COCO dataset."<<endl;
        while(true){
            if(waitKey(100)==13) break;
        }

        // make sure COCO_OutputPath exists
        fs::create_directories(COCO_OutputPath);
        cout<<"Output path: "<<COCO_OutputPath<<endl;

        // create a list of mask paths
        vector<fs::path> gray_mask_paths;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(COCORootPath/"stuffthingmaps_trainval2017/train2017"))
        {
            if(dir_entry.path().string().find(".png")!=string::npos){
                gray_mask_paths.push_back(dir_entry);
            }
        }
        
        // split all images to threads
        vector<vector<fs::path>> split_masks(numThreads);
        size_t t=0;
        fs::path voc_original_mask_path=VOCRootPath/"SegmentationClass";
        for (auto const& onegraymask : gray_mask_paths) 
        {
            if (t>numThreads-1) t=0;
            if(fs::exists(onegraymask)){
                split_masks[t].push_back(onegraymask);
            }
            t++;
        }
        cout<<"In total "<<gray_mask_paths.size()<<" original masks."<<endl;
        cout<<"Split for "<<split_masks.size()<<" threads. "<<endl;
        for (size_t i = 0; i < split_masks.size(); i++)
        {
            cout<<"[COCO] Thread "<<i<<": "<<split_masks[i].size()<<endl;
        }
        cout<<"files."<<endl;

        // multithread activation
        thread workers[numThreads-1];
        for (size_t i = 0; i < numThreads-1; i++)
        {
            workers[i]=thread(cocoimg2contrastive,split_masks[i+1],COCORootPath,COCO_OutputPath,COCO_OutputPath_binmask);
        }   
        cocoimg2contrastive(split_masks[0],COCORootPath,COCO_OutputPath,COCO_OutputPath_binmask,false);
        for (auto &one_thread : workers) one_thread.join();

        // write a filename list of all images
        vector<string> vec_AnchorFilename,vec_NanchorFilename;
        for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(COCO_OutputPath))
            {
                auto one_filename=dir_entry.path().filename();
                if (one_filename.string().find(".jpg")==string::npos) continue;
                if(one_filename.string().find("Nanchor")!=string::npos){
                    vec_NanchorFilename.push_back(one_filename);
                }
                else{
                    vec_AnchorFilename.push_back(one_filename);
                }
            }
        // sort filenames
        sort(vec_AnchorFilename.begin(),vec_AnchorFilename.end());
        sort(vec_NanchorFilename.begin(),vec_NanchorFilename.end());
        // write to .csv file
        ofstream ImgList;
        ImgList.open(COCO_OutputPath/"ImgList.csv");
        // header
        ImgList<<"anchor,nanchor\n";
        for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
        {
            ImgList<<vec_AnchorFilename[i]<<","<<vec_NanchorFilename[i]<<"\n";
        }
        ImgList.close();    
    }
    return 0;
}

void vocimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir){
    // Following voc_colormap is from
    // https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
    // black background is removed
    vector<vector<int>> voc_colormap={
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
    auto jpegPath=voc_root/"JPEGImages";
    for(auto const& OneColorfulMask: ColorfulMasks){
        auto corres_jpeg=jpegPath/(OneColorfulMask.stem().string()+".jpg");

        if (!fs::exists(corres_jpeg)){
            cout<<"File "<<corres_jpeg<<" does not exist."<<endl;
            abort();
        }
        Mat jpeg=imread(corres_jpeg);
        Mat tmp_mask=imread(OneColorfulMask);
        cvtColor(tmp_mask,tmp_mask,COLOR_BGR2RGB);

        vector<vector<Vector2i>> tmp_pixel_class(voc_colormap.size());// we would ignore black background and object edge
        // cout<<"Start pixel-wise match."<<endl;
        for (size_t r = 0; r < tmp_mask.rows; r++)
        {
            for (size_t c = 0; c < tmp_mask.cols; c++)
            {
                auto pixelRGB=tmp_mask.at<Vec3b>(r,c);
                vector<int> tmp_RGB={pixelRGB[0],pixelRGB[1],pixelRGB[2]};
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
            else if(tmp_pixel_class[i].size()<=0.2*tmp_mask.rows*tmp_mask.cols){
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
                    string bin_mask_filename=binmask_output_dir/(OneColorfulMask.stem().string()+"_binmask"+to_string(i)+".jpg");
                    string nbin_mask_filename=binmask_output_dir/(OneColorfulMask.stem().string()+"_nbinmask"+to_string(i)+".jpg");
                    imwrite(bin_mask_filename,tmp_bin_mask);
                    imwrite(nbin_mask_filename,~tmp_bin_mask);
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
            string anchor_filename=output_dir/(OneColorfulMask.stem().string()+"_anchor"+to_string(i)+".jpg");
            string Nanchor_filename=output_dir/(OneColorfulMask.stem().string()+"_Nanchor"+to_string(i)+".jpg");
            // cout<<"before bitwise_and."<<endl;
            bitwise_and(jpeg,bin_masks[i],tmp_anchor);
            // cout<<"between bitwise_and."<<endl;
            bitwise_and(jpeg,invert_bin_mask,tmp_Nanchor);
            // cout<<"after bitwise_and."<<endl;
            imwrite(anchor_filename,tmp_anchor);
            imwrite(Nanchor_filename,tmp_Nanchor);
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
    for (auto const& OneGrayMask:GrayscaleMasks)
    {
        auto corres_jpg=RawImagePath/(OneGrayMask.stem().string()+".jpg");
        if (!fs::exists(corres_jpg)){
            cout<<"File "<<corres_jpg<<" does not exist."<<endl;
            abort();
        }
        Mat jpeg=imread(corres_jpg,IMREAD_COLOR);
        Mat tmp_mask=imread(OneGrayMask,IMREAD_GRAYSCALE );

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
            else if(tmp_pixel_class[i].size()<=0.2*tmp_mask.rows*tmp_mask.cols){
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
                    string bin_mask_filename=binmask_output_dir/(OneGrayMask.stem().string()+"_binmask"+to_string(i)+".jpg");
                    string nbin_mask_filename=binmask_output_dir/(OneGrayMask.stem().string()+"_nbinmask"+to_string(i)+".jpg");
                    imwrite(bin_mask_filename,tmp_bin_mask);
                    imwrite(nbin_mask_filename,~tmp_bin_mask);
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
            string anchor_filename=output_dir/(OneGrayMask.stem().string()+"_anchor"+to_string(i)+".jpg");
            string Nanchor_filename=output_dir/(OneGrayMask.stem().string()+"_Nanchor"+to_string(i)+".jpg");
            // cout<<"before bitwise_and."<<endl;
            bitwise_and(jpeg,bin_masks[i],tmp_anchor);
            // cout<<"between bitwise_and."<<endl;
            bitwise_and(jpeg,invert_bin_mask,tmp_Nanchor);
            // cout<<"after bitwise_and."<<endl;
            imwrite(anchor_filename,tmp_anchor);
            imwrite(Nanchor_filename,tmp_Nanchor);
        }
    }    
}

void adeimg2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir, bool print_process){
    
}