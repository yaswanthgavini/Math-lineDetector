/*
written on 25/04/2015.
Binarization by global thresholding.
connected componets lableing and estimation of bounding boxes.
27/04/2015
Line Segmentation.
linsegment clustering
word segmentation.
spatial lableing.
feature extraction of ccs

04/05/2015
2pass is adjusted to end of the line
adding small lines to near lines.

*/

#include <cv.h>
#include <highgui.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define WTRSHOLD 50
#define GTRESHOLD 140
using namespace cv;
using namespace std;

void Findcc (const Mat & binary, vector < std::vector < Point2i > >&blobs);
int ComputeBbox (Mat &, Mat &);
void boxprint (Mat &, Mat &);
void lineprint (Mat &, Mat &);
double histDF (Mat &, Mat &, Rect &, int);
int sort (int[], int);
int median (int[], int);

int
main (int argc, char **argv)
{
  Mat img = imread (argv[1], 0);	// force greyscale
  if (!img.data)
    {
      std::cout << "File not found" << std::endl;
      return -1;
    }


  Mat output = Mat::ones (img.size (), CV_8UC3);
  output.setTo (Scalar (255, 255, 255));

  Mat binary;
  vector < vector < Point2i > >blobs;

  threshold (img, binary, GTRESHOLD, 255, THRESH_BINARY);
  Mat sub_mat = Mat::ones (binary.size (), binary.type ()) * 255;
  imwrite ("1bin.jpg", binary);
  subtract (sub_mat, binary, binary);
  imwrite ("2ibin.jpg", binary);

  Findcc (binary, blobs);
  Mat labledcc = Mat::ones (img.size (), CV_32SC1);
  int ccnum = 2;		// 1 for background
  // Randomy color allocation for blobs
  for (size_t i = 0; i < blobs.size (); i++)
    {
      unsigned char r = 255 * (rand () / (1.0 + RAND_MAX));
      unsigned char g = 255 * (rand () / (1.0 + RAND_MAX));
      unsigned char b = 255 * (rand () / (1.0 + RAND_MAX));

      for (size_t j = 0; j < blobs[i].size (); j++)
	{
	  int x = blobs[i][j].x;
	  int y = blobs[i][j].y;
	  output.at < Vec3b > (y, x)[0] = b;
	  output.at < Vec3b > (y, x)[1] = g;
	  output.at < Vec3b > (y, x)[2] = r;
	  labledcc.at < int >(y, x) = ccnum;

	}
      ++ccnum;
    }
  imwrite ("3out.jpg", output);
// noise removal

  vector < int >ccSize (blobs.size () + 2);
  for (int i = 0; i < img.rows; i++)
    {
      for (int j = 0; j < img.cols; j++)
	{
	  int t = labledcc.at < int >(i, j);
	  ccSize[t] += 1;
	}
    }

  int iccno = 0;
  for (int i = 0; i < blobs.size () + 2; i++)
    if (ccSize[i] < 25)
      ++iccno;

  printf ("\t\t%d %d %d\n", blobs.size (), ccnum, iccno);

  vector < int >noise (iccno);
  iccno = 0;
  for (int i = 0; i < blobs.size () + 2; i++)	// ccno 0 is nothing 
    if (ccSize[i] < 25)
      {
	noise[iccno] = i;
	++iccno;
      }

  for (int k = 0; k < iccno; k++)
    {
      for (int i = 0; i < img.rows; i++)
	{
	  for (int j = 0; j < img.cols; j++)
	    {
	      if (noise[k] == labledcc.at < int >(i, j))
		labledcc.at < int >(i, j) = 0;

	    }
	}
    }
  Mat clean = Mat::ones (binary.size (), binary.type ()) * 255;

  for (int i = 0; i < img.rows; i++)
    {
      for (int j = 0; j < img.cols; j++)
	{

	  if (labledcc.at < int >(i, j) > 1)
	    clean.at < uchar > (i, j) = 0;

	}
    }

  imwrite ("4clean.jpg", clean);
  Mat clean1 = clean.clone ();
// end of noise removal
  labledcc.release ();


// Re- allcating ccno 
  vector < vector < Point2i > >cblobs;
  subtract (sub_mat, clean1, clean1);
  Findcc (clean1, cblobs);
  Mat clabledcc = Mat::ones (img.size (), CV_32SC1);

  ccnum = 2;			// 1 for background

  for (size_t i = 0; i < cblobs.size (); i++)
    {
      for (size_t j = 0; j < cblobs[i].size (); j++)
	{
	  int x = cblobs[i][j].x;
	  int y = cblobs[i][j].y;
	  clabledcc.at < int >(y, x) = ccnum;

	}
      ++ccnum;
    }
  Mat bbox = Mat::zeros (cblobs.size (), 9, CV_32SC1);	//xmin,ymin,xmax,ymax,pixelcount,lineno,gx,gy,type
  ComputeBbox (clabledcc, bbox);

  Mat boximg, timg;
  threshold (clean, boximg, GTRESHOLD, 255, THRESH_BINARY);
  timg = boximg.clone ();
  cvtColor (boximg, boximg, CV_GRAY2RGB);
  boxprint (boximg, bbox);
  /* Rect r;
     for (int i = 0; i < bbox.rows; i++)
     {
     r.x = bbox.at < int >(i, 0);
     r.y = bbox.at < int >(i, 1);
     r.width = bbox.at < int >(i, 2) - bbox.at < int >(i, 0);
     r.height = bbox.at < int >(i, 3) - bbox.at < int >(i, 1);
     rectangle (boximg, r, Scalar (0, 0, 255), 1, 8, 0);


     } */
  imwrite ("5box.jpg", boximg);
// Linesegmentation is start here
  Mat fpass = timg.clone ();
//cout<<bbox;
  for (int i = 0; i < bbox.rows; i++)
    {
      int x1, y1, x2, y2;
      x1 = bbox.at < int >(i, 0);
      y1 = bbox.at < int >(i, 1);
      x2 = bbox.at < int >(i, 2);
      y2 = bbox.at < int >(i, 3);
      for (int j = x1 - 2; j < x2 + 2; j++)
	{
	  for (int k = y1 - 2; k < y2 + 2; k++)
	    {
	      fpass.at < uchar > (k, j) = 0;
	    }
	}


    }

  imwrite ("6fpass.jpg", fpass);

  Mat wpass = fpass.clone ();

  for (int i = 0; i < bbox.rows; i++)
    {
      int x1, y1, x2, y2;
      x1 = bbox.at < int >(i, 0);
      y1 = bbox.at < int >(i, 1);
      x2 = bbox.at < int >(i, 2);
      y2 = bbox.at < int >(i, 3);
      for (int j = x1 - 2; j < wpass.cols; j++)
	{
	  for (int k = y1 - 2; k < y2 + 2; k++)
	    {
	      wpass.at < uchar > (k, j) = 0;
	    }
	}
    }

  imwrite ("7wpass.jpg", wpass);



  vector < vector < Point2i > >lblobs;
  subtract (sub_mat, wpass, wpass);
  Findcc (wpass, lblobs);
  Mat labledline = Mat::ones (img.size (), CV_32SC1);

  ccnum = 2;			// 1 for background

  for (size_t i = 0; i < lblobs.size (); i++)
    {
      for (size_t j = 0; j < lblobs[i].size (); j++)
	{
	  int x = lblobs[i][j].x;
	  int y = lblobs[i][j].y;
	  labledline.at < int >(y, x) = ccnum;

	}
      ++ccnum;
    }
  Mat lbox = Mat::zeros (cblobs.size (), 4, CV_32SC1);	//xmin,ymin,xmax,ymax
  ComputeBbox (labledline, lbox);
// Display the line seg results


  Mat limg;
  threshold (clean, limg, GTRESHOLD, 255, THRESH_BINARY);
  //timg = limg.clone ();
  cvtColor (limg, limg, CV_GRAY2RGB);

  boxprint (limg, lbox);

  imwrite ("8line.jpg", limg);
//projection profile
  int count;
  Mat pp = Mat::ones (binary.size (), binary.type ()) * 255;
  imwrite ("9test.jpg", timg);
  for (int i = 0; i < boximg.rows; i++)
    {
      count = 0;
      for (int j = 0; j < boximg.cols; j++)
	{
	  if (timg.at < uchar > (i, j) == 0)
	    count++;
	}
      for (int k = 0; k < count; k++)
	pp.at < uchar > (i, k) = 0;
    }
  imwrite ("10pp.jpg", pp);


  Mat bpp = Mat::ones (binary.size (), binary.type ()) * 255;

  for (int i = 0; i < boximg.rows; i++)
    {
      count = 0;
      for (int j = 0; j < boximg.cols; j++)
	{
	  if (fpass.at < uchar > (i, j) == 0)
	    count++;
	}
      for (int k = 0; k < count; k++)
	bpp.at < uchar > (i, k) = 0;
    }
  imwrite ("11bpp.jpg", bpp);
//detection of small lines

  vector < vector < Point2i > >ppblobs;
  subtract (sub_mat, bpp, bpp);
  Findcc (bpp, ppblobs);
  Mat tlabledline = Mat::ones (img.size (), CV_32SC1);
  ccnum = 2;			// 1 for background

  for (size_t i = 0; i < ppblobs.size (); i++)
    {
      for (size_t j = 0; j < ppblobs[i].size (); j++)
	{
	  int x = ppblobs[i][j].x;
	  int y = ppblobs[i][j].y;
	  tlabledline.at < int >(y, x) = ccnum;

	}
      ++ccnum;
    }
  cout << "total lines is" << ppblobs.size () << " ";
  int col = boximg.cols / 20;
  cout << col << "\n";
  int b = 1;
  Mat sline = Mat::zeros (ppblobs.size (), 1, CV_32SC1);
  int lc = 0;
  for (int i = 0; i < boximg.rows; i++)
    {
      int a = tlabledline.at < int >(i, 10);
//cout<<a<<" ";
      if (a != 1)
	{

	  int flag = 0;
	  for (int j = 0; j < boximg.rows; j++)
	    {
	      if (tlabledline.at < int >(j, col) == a)
		{
		  flag = 1;
		  break;
		}

	    }

	  if (flag == 0 && (b != a || b == 1))
	    {
	      b = a;
	      sline.at < int >(lc, 0) = a;
	      ++lc;
	      printf ("line num %d is small\n", a);
	    }
	}
    }
  int tv, v, h, vcc, hcc, pcc1;
  v = h = vcc = hcc = tv = pcc1 = 0;
  int ccc, pcc, ncc;

  for (int j = 0; j < lc; j++)
    {
      int flag = 0;
      for (int i = 0; i < boximg.rows; i++)
	{
	  ccc = tlabledline.at < int >(i, 0);

	  if (tlabledline.at < int >(i, 0) == 1)
	    {
	      ++tv;


	    }
	  if (tlabledline.at < int >(i, 0) != sline.at < int >(j, 0) && tlabledline.at < int >(i, 0) > 1)
	    {
	      tv = 0;
	      pcc1 = tlabledline.at < int >(i, 0);

	    }
	  if (tlabledline.at < int >(i, 0) == sline.at < int >(j, 0) && flag == 0)
	    {
	      v = tv;
	      flag = 1;
	      pcc = pcc1;
	    }

	  if (flag == 1 && tlabledline.at < int >(i, 0) == 1)
	    {
	      ++h;
	    }
	  if (flag == 1 && tlabledline.at < int >(i, 0) > 1 && tlabledline.at < int >(i, 0) != sline.at < int >(j, 0))
	    {
	      flag = 2;
	      ncc = ccc;
	    }

	}
      printf ("dist v1 is %d v2 is %d\n", v, h);
      if (v > h)
	{
	  for (int i1 = 0; i1 < boximg.rows; i1++)
	    {
	      for (int j1 = 0; j1 < boximg.cols; j1++)
		{
		  if (tlabledline.at < int >(i1, j1) == sline.at < int >(j, 0))
		    tlabledline.at < int >(i1, j1) = ncc;
		}
	    }
	  //  printf ("%d is replaced with %d\n", sline.at < int >(j, 0), ncc);
	}
      if (v < h)
	{
	  for (int i1 = 0; i1 < boximg.rows; i1++)
	    {
	      for (int j1 = 0; j1 < boximg.cols; j1++)
		{
		  if (tlabledline.at < int >(i1, j1) == sline.at < int >(j, 0))
		    tlabledline.at < int >(i1, j1) = pcc;
		}
	    }
	  //  printf ("%d is replaced with %d\n", sline.at < int >(j, 0), pcc);
	}
    }
  Mat fline = Mat::zeros (ppblobs.size (), 4, CV_32SC1);	//xmin,ymin,xmax,ymax
  int flcount = ppblobs.size () - lc;
  ComputeBbox (tlabledline, fline);
  Mat flimg;
  threshold (clean, flimg, GTRESHOLD, 255, THRESH_BINARY);
  cvtColor (flimg, flimg, CV_GRAY2RGB);
  lineprint (flimg, fline);
  imwrite ("12line.jpg", flimg);


  Mat ulimg = timg.clone ();
//cout<<bbox;
  Rect r1;
  for (int i = 0; i < fline.rows; i++)
    {
      if (fline.at < int >(i, 0) > 5000 || fline.at < int >(i, 2) < -9)
	continue;
      r1.x = fline.at < int >(i, 0);
      r1.y = fline.at < int >(i, 1);
      r1.width = limg.rows;
      r1.height = fline.at < int >(i, 3) - fline.at < int >(i, 1);

      for (int j = r1.y; j < fline.at < int >(i, 3); j++)
	{
	  for (int k = r1.x; k < r1.width; k++)
	    {
	      ulimg.at < uchar > (j, k) = 0;
	    }
	}


    }

  imwrite ("13ulimg.jpg", ulimg);


//for serial line

  vector < vector < Point2i > >lblobs9;
  subtract (sub_mat, ulimg, ulimg);
  Findcc (ulimg, lblobs9);
  Mat labledline9 = Mat::ones (img.size (), CV_32SC1);

  ccnum = 2;			// 1 for background

  for (size_t i = 0; i < lblobs9.size (); i++)
    {
      for (size_t j = 0; j < lblobs9[i].size (); j++)
	{
	  int x = lblobs9[i][j].x;
	  int y = lblobs9[i][j].y;
	  labledline9.at < int >(y, x) = ccnum;

	}
      ++ccnum;
    }
  Mat lbox9 = Mat::zeros (lblobs9.size (), 4, CV_32SC1);	//xmin,ymin,xmax,ymax
  ComputeBbox (labledline9, lbox9);
  Mat limg9 = timg.clone ();
  cvtColor (limg9, limg9, CV_GRAY2RGB);
  boxprint (limg9, lbox9);

  imwrite ("14ulimg.jpg", limg9);

//feature genaration

  Mat feat = Mat::zeros (lblobs9.size (), 2, CV_32FC1);
  Mat label = Mat::zeros (lblobs9.size (), 1, CV_32SC1);
//Mat temp=Mat::zeros(lblobs9.size(), 1, CV_32SC1);
  Mat temp1 = Mat::zeros (lblobs9.size (), 1, CV_32SC1);
  Mat temp2 = Mat::zeros (lblobs9.size (), 1, CV_32SC1);
  Mat temp3 = Mat::zeros (lblobs9.size (), 1, CV_32SC1);
  Mat temp;
  int array[lblobs9.size ()];
  for (int i = 0; i < lblobs9.size (); i++)
    {
      feat.at < float >(i, 0) = (float)(lbox9.at < int >(i, 3) - lbox9.at < int >(i, 1));
      array[i] = lbox9.at < int >(i, 3) - lbox9.at < int >(i, 1);
      temp1.at<int>(i,0) = lbox9.at < int >(i, 3) - lbox9.at < int >(i, 1);
      feat.at < float >(i, 1) = (float)(lbox9.at < int >(i, 1) - lbox9.at < int >(i - 1, 3));
      if (i == 0)
	feat.at < float >(i, 1) = 5.0;

 //     feat.at < float >(i, 2) = (float)(lbox9.at < int >(i + 1, 1) - lbox9.at < int >(i, 3));
//      if (i == lblobs9.size () - 1)
//	feat.at < float >(i, 2) = 5.0;
temp2.at<int>(i,0)=(int)feat.at < float >(i, 1);
//temp3.at<int>(i,0)=(int)feat.at < float >(i, 2);

    }
  //cout << temp1 << endl;
//sortIdx(temp1, temp, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
//sort(temp1, temp, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
 // cout << temp << endl;


  int median = sort (array, lblobs9.size ());
  cout << median<<endl;
// Initialigetion with a guess label have 3 clusters graphics, math, normal
for (int i = 0; i < lblobs9.size (); i++)
    {
label.at<int>(i,0)=0;
if(feat.at < float >(i, 0)>(1.5*median) && (4*median) >feat.at < float >(i, 0))label.at<int>(i,0)=1;
if((4*median) < feat.at < float >(i, 0))label.at<int>(i,0)=2;
//cout<<label.at<int>(i,0)<<" ";
}
cout<<label<<endl;
Mat centers;
//kmeans(feat, 3, label, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),3, KMEANS_USE_INITIAL_LABELS, centers);
cout<<temp1<<endl;
cout<<temp2<<endl;
//cout<<temp3<<endl;
cout<<label<<endl;

// final output

Mat final1=binary.clone();
  subtract (sub_mat, final1, final1);
  cvtColor (final1, final1, CV_GRAY2RGB);

//Rect r1;
  for (int i = 0; i < lbox9.rows; i++)
    {
      
      r1.x = lbox9.at < int >(i, 0);
      r1.y = lbox9.at < int >(i, 1);
      r1.width = lbox9.at < int >(i, 2) - lbox9.at < int >(i, 0);
      r1.height = lbox9.at < int >(i, 3) - lbox9.at < int >(i, 1);
	if(label.at < int >(i, 0)==0)
      rectangle (final1, r1, Scalar (0, 255, 0), 3, 8, 0);
	if(label.at < int >(i, 0)==1)
      rectangle (final1, r1, Scalar (255, 0, 0), 3, 8, 0);
if(label.at < int >(i, 0)==2)
      rectangle (final1, r1, Scalar (0, 0, 255), 3, 8, 0);
    }
  imwrite ("0final.jpg", final1);
  /* Mat tmp, tmp1;
     threshold (clean, tmp, GTRESHOLD, 255, THRESH_BINARY);
     //cvtColor (tmp1, tmp, CV_RGB2GRAY);
     Mat boximg2 = imread ("6fpass.jpg");
     Mat boximg1;
     cvtColor(boximg2,boximg1,CV_BGR2GRAY,1);
     //cvtColor (boximg2, boximg1, CV_RGB2GRAY);
     Rect r1;
     int lc1 = 1;

     for (int i = 0; i < fline.rows; i++)
     {
     if (fline.at < int >(i, 0) < 8000 && fline.at < int >(i, 1) < 8000 && fline.at < int >(i, 2) > -9 && fline.at < int >(i, 3) > -9
     && (fline.at < int >(i, 3) - fline.at < int >(i, 1))>6)
     {
     r1.x = 0;
     r1.y = fline.at < int >(i, 1);
     r1.width = tmp.cols;
     r1.height = fline.at < int >(i, 3) - fline.at < int >(i, 1);

     histDF (boximg1, tmp, r1, lc1);
     ++lc1;

     }
     } */
}


void
Findcc (const Mat & binary, std::vector < std::vector < Point2i > >&blobs)
{
  blobs.clear ();

  // Fill the label_image with the blobs
  // 0  - background
  // 1  - unlabelled foreground
  // 2+ - labelled foreground

  Mat label_image;
  binary.convertTo (label_image, CV_32SC1);

  int label_count = 2;		// starts at 2 because 0,1 are used already

  for (int y = 0; y < label_image.rows; y++)
    {
      int *row = (int *) label_image.ptr (y);
      for (int x = 0; x < label_image.cols; x++)
	{
	  if (row[x] != 255)
	    {
	      continue;
	    }

	  Rect rect;
	  floodFill (label_image, Point (x, y), label_count, &rect, 0, 0, 4);
	  vector < Point2i > blob;
	  for (int i = rect.y; i < (rect.y + rect.height); i++)
	    {
	      int *row2 = (int *) label_image.ptr (i);
	      for (int j = rect.x; j < (rect.x + rect.width); j++)
		{
		  if (row2[j] != label_count)
		    {
		      continue;
		    }

		  blob.push_back (Point2i (j, i));
		}
	    }
	  blobs.push_back (blob);
	  label_count++;
	}
    }
}

int
ComputeBbox (Mat & rlsaImg, Mat & mbr)
{
  //printf ("%d/n", mbr.rows);

  for (int k = 0; k < mbr.rows; k++)
    {
      mbr.at < int >(k, 0) = 10000;
      mbr.at < int >(k, 1) = 10000;
      mbr.at < int >(k, 2) = -999;
      mbr.at < int >(k, 3) = -999;


    }

  int i, j, s, value, base = 0;

  for (i = 0; i < rlsaImg.cols; i++)
    {
      for (j = 0; j < rlsaImg.rows; j++)
	{
	  if (rlsaImg.at < int >(j, i) > 1)
	    {
	      value = rlsaImg.at < int >(j, i) - 2;


	      if (mbr.at < int >(value, 0) > (base + i))
		mbr.at < int >(value, 0) = base + i;
	      if (mbr.at < int >(value, 1) > j)
		mbr.at < int >(value, 1) = j;
	      if (mbr.at < int >(value, 2) < (base + i))
		mbr.at < int >(value, 2) = base + i;
	      if (mbr.at < int >(value, 3) < j)
		mbr.at < int >(value, 3) = j;
	    }
	}
    }

  //cout << mbr;
/*
 int numcomp = mbr.rows;

  for (i = 1; i <= (numcomp); i++)
    {
      if (( (mbr.at < int >(i, 2) - mbr.at < int >(i, 0)) * (mbr.at < int >(i, 3) - mbr.at < int >(i, 1)) ) < 25)
	{
	  for (j = i + 1; j <= (numcomp); j++)
	    {
	      mbr.at < int >(j - 1, 2) = mbr.at < int >(j, 2);
	      mbr.at < int >(j - 1, 0) = mbr.at < int >(j, 0);
	      mbr.at < int >(j - 1, 3) = mbr.at < int >(j, 3);
	      mbr.at < int >(j - 1, 1) = mbr.at < int >(j, 1);
	    }
	  (numcomp)--;
	  i--;
	}
    }

  for (j = 0; j < (mbr.rows - numcomp); j++)
    {
      mbr.at < int >(mbr.rows - j, 2) = 0;
      mbr.at < int >(mbr.rows - j, 0) = 0;
      mbr.at < int >(mbr.rows - j, 3) = 0;
      mbr.at < int >(mbr.rows - j, 1) = 0;

    }

*/

}

void
boxprint (Mat & limg, Mat & lbox)
{
  Rect r1;
  for (int i = 0; i < lbox.rows; i++)
    {
      if (lbox.at < int >(i, 0) > 5000 || lbox.at < int >(i, 2) < -9)
	continue;
      r1.x = lbox.at < int >(i, 0);
      r1.y = lbox.at < int >(i, 1);
      r1.width = lbox.at < int >(i, 2) - lbox.at < int >(i, 0);
      r1.height = lbox.at < int >(i, 3) - lbox.at < int >(i, 1);
      rectangle (limg, r1, Scalar (0, 0, 255), 1, 8, 0);


    }
}

void
lineprint (Mat & limg, Mat & lbox)
{
  Rect r1;
  for (int i = 0; i < lbox.rows; i++)
    {
      if (lbox.at < int >(i, 0) > 5000 || lbox.at < int >(i, 2) < -9)
	continue;
      r1.x = lbox.at < int >(i, 0);
      r1.y = lbox.at < int >(i, 1);
      r1.width = limg.rows;
      r1.height = lbox.at < int >(i, 3) - lbox.at < int >(i, 1);
      rectangle (limg, r1, Scalar (0, 0, 255), 1, 8, 0);


    }
}

double
histDF (Mat & limg, Mat & bimg, Rect & r1, int lno)
{				/*
				   for (int i = r1.y; i < r1.y + r1.height; i++)
				   {

				   for (int j = 0; j < r1.width; j++)
				   {
				   printf ("%d ", bimg.at < uchar > (i, j));
				   } printf ("\n");
				   } */


  cout << limg.depth () << ", " << limg.channels () << endl;
  cout << limg.type () << "," << bimg.type () << endl;
  cout << bimg.depth () << ", " << bimg.channels () << endl;
  //if (lbox.at < int >(i, 0) == 10000 && lbox.at < int >(i, 2) == -999)
  // return -9;

  // rectangle (limg, r1, Scalar (0, 0, 255), 1, 8, 0);
  printf ("Rect is %d %d %d %d\n", r1.x, r1.y, r1.width, r1.height);
  // Mat roi1 = bimg (r1);
  //Mat roi2 = limg (r1);
  //char name[500];
  //sprintf (name, "./tmp/bline%d.jpg", lno);
  // Mat roi = bimg ( Rect(r1.x,r1.y,r1.height-2,r1.width-2));
  //Mat roi = bimg ( Rect(0,1800,800,800));
  //imwrite (name, roi1);
  //sprintf (name, "./tmp/line%d.jpg", lno);
  //imwrite (name, roi2);
  imwrite ("./tmp/bline.jpg", bimg);
  imwrite ("./tmp/line.jpg", limg);
  Mat hist_bimg = Mat::zeros (Size (r1.height, 1), CV_32FC1);
  Mat hist_limg = Mat::zeros (Size (r1.height, 1), CV_32FC1);
  int bc, lc;

  int i1 = 0;
  int t9;

  for (int i = r1.y; i < (r1.y + r1.height); i++)
    {
      bc = lc = 0.0;
      for (int j = 0; j < r1.width; j++)
	{

	  if (bimg.at < uchar > (i, j) < 255)
	    {
	      bc++;
	    }
	  if (limg.at < uchar > (i, j) < 255)
	    {
	      ++lc;
	    }

	  // printf("%d %d %d %d \n ",i,j, limg.at < uchar >(i, j),lc);

	}
//scanf("%d",&t9);

//printf("%f %f\n",bc,lc);

      hist_bimg.at < float >(i1, 0) = (float) bc;
      hist_limg.at < float >(i1, 0) = (float) lc;
//printf(" %f %f %d\n",(float)lc,hist_limg.at < float >(i1, 0),i1);
      lc = 0;
      i1++;
    }
  // cout << hist_bimg << "\n" << hist_limg << "\n";
  int histSize = r1.height;
  for (int i = 0; i < r1.height; i++)
    {
      cout << (float) hist_limg.at < float >(i, 0) << " ";

    }
  cout << endl;
  for (int i = 0; i < r1.height; i++)
    {
      cout << (float) hist_bimg.at < float >(i, 0) << " ";

    }
  cout << endl;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, r1.width };
  const float *histRange = { range };

  int compare_method = 3;
  // double base_base = compareHist (hist_bimg, hist_limg, compare_method);
  Mat Res;
  matchTemplate (hist_bimg, hist_limg, Res, CV_TM_CCORR_NORMED);
  // printf ("corelation is %f\n", base_base);
  cout << Res << endl;
}



int
median (int new_array[], int num)
{
  //CALCULATE THE MEDIAN (middle number)
  if (num % 2 != 0)
    {				// is the # of elements odd?
      int temp = ((num + 1) / 2) - 1;
      //cout << "The median is " << new_array[temp] << endl;
      return new_array[temp];
    }
  else
    {				// then it's even! :)
     // cout << "The median is " << new_array[(num / 2) - 1] << " and " << new_array[num / 2] << endl;
      return (new_array[(num / 2) - 1] + new_array[num / 2]) / 2;
    }

}

int
sort (int new_array[], int num)
{
  //ARRANGE VALUES
  for (int x = 0; x < num; x++)
    {
      for (int y = 0; y < num - 1; y++)
	{
	  if (new_array[y] > new_array[y + 1])
	    {
	      int temp = new_array[y + 1];
	      new_array[y + 1] = new_array[y];
	      new_array[y] = temp;
	    }
	}
    }

  
  return median (new_array, num);
}
