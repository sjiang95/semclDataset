#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<filesystem>
#include<string>
#include<thread>
#include <fstream>

#include<eigen3/Eigen/Dense>

using namespace cv;
using namespace std;
namespace fs=std::filesystem;
using namespace Eigen;

void img2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir);

// Following colormap is from
// https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
// black background is removed
vector<vector<int>> colormap={
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
    string mode="semantic";
    bool write_binmask=false;
    // If there is input argument.
    if (argc!=1){
        for (size_t i = 1; i < argc; i=i+2)
        {
            if(string("--voc_path").compare(argv[i])==0){
                VOCRootPath=argv[i+1];
                cout<<"Given VOCRootPath: "<<VOCRootPath<<endl;
                continue;
            }
            else if(string("--mode").compare(argv[i])==0){
                if (string(argv[i+1]).compare("semantic")==0 || string(argv[i+1]).compare("instance")==0){
                    mode=argv[i+1];
                    continue;
                }
                else{
                    cout<<"Set mode as either semantic or instance."<<endl;
                    return -1;
                }
            }
            else if(string("--save_binmask").compare(argv[i])==0){
                write_binmask=true;
            }
            else{
                cout<<"Unknown option: "<<argv[i]<<endl;
                return -1;
            }
        }               
    }

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

    fs::path OutputSurfix="ContrastivePairs_"+mode;
    fs::path OutputSurfix_binmask="ContrastivePairs_"+mode+"_binmask";
    auto OutputPath=fs::current_path()/OutputSurfix;
    auto OutputPath_binmask=fs::current_path()/OutputSurfix_binmask;
    cout<<"Attempt to use VOC dataset path: "<<VOCRootPath<<endl;
    if (write_binmask) {
        fs::create_directory(OutputPath_binmask);
        cout<<"Binary masks will be saved to: "<<OutputPath_binmask<<endl;
    }
    else{
        OutputPath_binmask= fs::path(OutputPath_binmask.string().erase());
    }

    // make sure OutputPath exists
    fs::create_directory(OutputPath);
    cout<<"Output path: "<<OutputPath<<endl;

    fs::path voc_original_mask_path;
    if (mode.compare("semantic")==0){
        voc_original_mask_path=VOCRootPath/"SegmentationClass";
    }
    else if (mode.compare("semantic")==0){
        voc_original_mask_path=VOCRootPath/"SegmentationObject";
    }

    fs::path train_set_txt=VOCRootPath/"ImageSets/Segmentation/train.txt";
    ifstream txt;
    txt.open(train_set_txt);
    vector<string> train_set_filename;
    assert(txt.is_open());
    string tmp_txt;
    while(getline(txt,tmp_txt)){
        train_set_filename.push_back(tmp_txt);
    }

    vector<fs::path> voc_original_masks;
    vector<vector<fs::path>> split_masks(numThreads);
    size_t t=0;
    for (auto const& onefilename : train_set_filename) 
    {
        if (t>15) t=0;
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
        cout<<"Thread "<<i<<": "<<split_masks[i].size()<<endl;
    }
    cout<<"files."<<endl;

    // multithread activation
    thread workers[numThreads-1];
    for (size_t i = 0; i < numThreads-1; i++)
    {
        workers[i]=thread(img2contrastive,split_masks[i+1],VOCRootPath,OutputPath,OutputPath_binmask);
    }   
    img2contrastive(split_masks[0],VOCRootPath,OutputPath,OutputPath_binmask);
    for (auto &one_thread : workers) one_thread.join();

    // write a filename list of all images
    vector<string> vec_AnchorFilename,vec_NanchorFilename;
    for (const fs::directory_entry& dir_entry : std::filesystem::recursive_directory_iterator(OutputPath))
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
    ImgList.open(OutputPath/"ImgList.csv");
    // header
    ImgList<<"anchor,nanchor\n";
    for (size_t i = 0; i < vec_AnchorFilename.size(); i++)
    {
        ImgList<<vec_AnchorFilename[i]<<","<<vec_NanchorFilename[i]<<"\n";
    }
    ImgList.close();    

    return 0;
}

void img2contrastive(vector<fs::path> ColorfulMasks, fs::path voc_root, fs::path output_dir, fs::path binmask_output_dir){
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

        vector<vector<Vector2i>> tmp_pixel_class(colormap.size());// we would ignore black background and object edge
        // cout<<"Start pixel-wise match."<<endl;
        for (size_t r = 0; r < tmp_mask.rows; r++)
        {
            for (size_t c = 0; c < tmp_mask.cols; c++)
            {
                auto pixelRGB=tmp_mask.at<Vec3b>(r,c);
                vector<int> tmp_RGB={pixelRGB[0],pixelRGB[1],pixelRGB[2]};
                Vector2i coordinates(r,c);
                for (size_t i = 0; i < colormap.size(); i++)
                {
                    if(colormap[i]==tmp_RGB){
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
                    imwrite(bin_mask_filename,tmp_bin_mask);
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