/////////////////////////////////////////////////////////
//
//       ■■■■                            ■■■■
//     ■        ■    ■            ■    ■        ■
//     ■        ■        ■    ■        ■        ■
//     ■        ■    ■  ■    ■  ■    ■        ■
//     ■    ■  ■    ■  ■    ■  ■    ■    ■  ■
//     ■      ■■    ■  ■    ■  ■    ■      ■■
//       ■■■■  ■  ■    ■■    ■      ■■■■  ■
//
////////////////////////////////////////////////////////
#include "stdafx.h"
#include "stdlib.h"
#include "math.h"
#include <iostream>
#include <fstream>
using namespace std;

#define COL              28
#define ROW              28
#define LR             0.01
#define NUM              10
#define LAYERS            2
#define LAYER1           10
#define LAYER2           10
#define PIXEL           784
#define TRAIN_DATA    60000
#define TEST_DATA     10000
#define TRAINING_TIME    1

//784 * 10 * 10

class BP
{
public:
	char img[PIXEL];
	double w0[LAYER1][PIXEL];
	double w1[LAYER2][LAYER1];
	double ans[NUM];
	double value[2][NUM];
	double err[2][NUM];

	BP()
	{
		for (int i = 0; i < PIXEL; i++) img[i] = 0;

		for (int i = 0; i < NUM; i++)
		for (int j = 0; j < PIXEL; j++) w0[i][j] = 0;

		for (int i = 0; i < NUM; i++)
		for (int j = 0; j < NUM; j++) w1[i][j] = 0;

		for (int i = 0; i < 2; i++)
		for (int j = 0; j < NUM; j++) value[i][j] = 0;

		for (int i = 0; i < NUM; i++) ans[i] = 0;

		for (int i = 0; i < 2; i++)
		for (int j = 0; j < NUM; j++) err[i][j] = 0;
	}

	void readMNIST_TrainData()
	{
		char c;
		FILE *f;
		char a;
		FILE *ft;

		f = fopen("train-images.idx3-ubyte", "rb");

		for (int i = 0; i < 16; i++) fscanf(f, "%c", &c);

		ft = fopen("train-labels.idx1-ubyte", "r");
		for (int i = 0; i < 8; i++) fscanf(ft, "%c", &a);

		for (int z = 0; z < TRAIN_DATA; z++)
		{
			for (int i = 0; i < PIXEL; i++)
			{
				fscanf(f, "%c", &c);
				if (c < 0) img[i] = c*-1;
				else img[i] = c;
			}

			for (int j = 0; j < NUM; j++) ans[j] = 0;
			fscanf(ft, "%c", &a);
			ans[(int)a] = 1;

			for (int j = 0; j < LAYERS; j++) training(j);

			getErr();
			adjustWeight();

			
		}
	}

	void training(int n)
	{
		if (n == 0)
		for (int i = 0; i < LAYER1; i++)
		{
			double sum = 0;
			for (int j = 0; j < PIXEL; j++)
			{
				sum += w0[i][j] * (double)img[j] / 255;
			}
			value[n][i] = 1.0 / (1.0 + exp(-1.0 * sum));
		}
		else
		for (int i = 0; i < LAYER2; i++)
		{
			double sum = 0;
			for (int j = 0; j < LAYER1; j++)
			{
				sum += w1[i][j] * value[n - 1][j];
			}
			value[n][i] = 1.0 / (1.0 + exp(-1.0 * sum));
		}
	}

	void getErr()
	{
		for (int i = 0; i < LAYER2; i++)
		{
			err[1][i] = ans[i] - value[1][i];
		}

		for (int i = 0; i < LAYER2; i++)
		{
			double d = 0;
			for (int j = 0; j < LAYER1; j++) d += err[1][i] * w1[i][j];
			err[0][i] = d * value[0][i] * (1 - value[0][i]);
		}
	}

	void adjustWeight()
	{
		for (int i = 0; i < LAYER1; i++)
		{
			for (int j = 0; j < PIXEL; j++)
			{
				w0[i][j] += LR * err[0][i] * (double)img[j] / 255;
			}
		}

		for (int i = 0; i < LAYER2; i++)
		{
			for (int j = 0; j < LAYER1; j++)
			{
				w1[i][j] += LR * err[1][i] * value[0][j];
			}
		}
	}

	void readTest()
	{
		char c;
		FILE *f;
		char a;
		FILE *ft;
		int correct = 0;

		f = fopen("t10k-images.idx3-ubyte", "rb");

		for (int i = 0; i < 16; i++) fscanf(f, "%c", &c);

		ft = fopen("t10k-labels.idx1-ubyte", "r");
		for (int i = 0; i < 8; i++) fscanf(ft, "%c", &a);

		for (int z = 0; z < TEST_DATA; z++)
		{
			for (int i = 0; i < PIXEL; i++)
			{
				fscanf(f, "%c", &c);
				if (c < 0) img[i] = c*-1;
				else img[i] = c;
			}

			for (int j = 0; j < NUM; j++) ans[j] = 0;
			fscanf(ft, "%c", &a);
			ans[(int)a] = 1;
			
			for (int j = 0; j < LAYERS; j++) goTest(j);

			double MAX = 0;
			int location = 0;
			for (int j = 0; j < LAYER2; j++) 
			{
				if (value[1][j]>MAX)
				{
					MAX = value[1][j];
					location = j;
				}
			}
			if (location == (int)a) correct++;
		}
		printf("測試結果 : %f\n", (double)correct / TEST_DATA);
	}

	void goTest(int n)
	{
		if (n==0)
		for (int i = 0; i < LAYER1; i++)
		{
			double sum = 0;
			for (int j = 0; j < PIXEL; j++)
			{
				sum += w0[i][j] * (double)img[j] / 255;
			}
			value[n][i] = 1.0 / (1.0 + exp(-1.0 * sum));
		}
		else
		for (int i = 0; i < LAYER2; i++)
		{
			double sum = 0;
			for (int j = 0; j < LAYER1; j++)
			{
				sum += w1[i][j] * value[n - 1][j];
			}
			value[n][i] = 1.0 / (1.0 + exp(-1.0 * sum));
		}
	}
};


int _tmain(int argc, _TCHAR* argv[])
{
	BP bp;
	for (int i = 0; i < TRAINING_TIME; i++) 
	{
		printf("訓練次數 : %d / %d\n", i+1, TRAINING_TIME);
		bp.readMNIST_TrainData();
	}
	bp.readTest();

	return 0;
}

