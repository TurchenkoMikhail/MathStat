#include <iostream>
#include <cmath>
#include <random>
using std::cout;
using std::random_device;
using std::mt19937;
using std::normal_distribution;
using std::cauchy_distribution;
using std::poisson_distribution;
using std::uniform_real_distribution;

typedef enum {
  NORMAL,
  CAUCHY,
  POISSON,
  UNIFORM, 

  COUNT
}sample_t;

#define PI 3.1415926

double FindX14(std::vector<double>& sortedData);
double FindX34(std::vector<double>& sortedData);
double FindX(std::vector<double>& data);
double FindDX(std::vector<double>& data);
double FindMedX(std::vector<double>& sortedData);
double FindzR(std::vector<double>& sortedData);
double FindzQ(std::vector<double>& sortedData);
double Findztr(std::vector<double>& sortedData);

typedef std::vector<double> vec;

//gaussian kernel
double K1(double u) {
  return exp(-(u * u)/2.0) / sqrt(2.0 * PI);
}

/*
//cauchy
double K2(double u) {
  return 1.0 /(PI*(u * u + 1));
}

//uniform
double K4(double u) { return 0.5; }
*/



//1. Сгенерировать выборки размером 10, 50 и 1000 элементов.
//Построить на одном рисунке гистограмму и график плотности распределения
void Step1() {
  double sqrt3 = sqrt(3.0);
  random_device rd{};
  mt19937 gen{ rd() };
  normal_distribution<> normal(0, 1);
  cauchy_distribution<> cauchy(0, 1);
  poisson_distribution<> poisson(10.0);
  uniform_real_distribution<> uniform(-sqrt3, sqrt3);

  const int sizeOfSample[] = { 10, 50, 1000 }, size = 3;

  sample_t sample = NORMAL;
  //sample_t sample = CAUCHY;
  //sample_t sample = POISSON;
  //sample_t sample = UNIFORM

  if (sample == NORMAL) cout << "Normal:";
  else if (sample == CAUCHY) cout << "Cauchy:";
  else if (sample == POISSON) cout << "Poisson:";
  else cout << "Uniform:";
  cout <<"\n";
  
  for (int i = 0; i < size; ++i) {
    cout << "Size = " << sizeOfSample[i] << "\n";
    for (int j = 0; j < sizeOfSample[i]; ++j) {
      if(sample==NORMAL) cout << normal(gen);
      else if (sample == CAUCHY) cout << cauchy(gen);
      else if (sample == POISSON) cout << poisson(gen);
      else cout << uniform(gen);
      cout << " ";
    }
    cout << "\n\n";
  }

}

//2. Сгенерировать выборки размером 10, 100 и 1000 элементов.
//Для каждой выборки вычислить следующие статистические характеристики положения данных : <x>, 𝑚𝑒𝑑 𝑥, 𝑧𝑅, 𝑧𝑄, 𝑧𝑡𝑟.Повторить такие
//вычисления 1000 раз для каждой выборки и найти среднее характеристик положения и их квадратов :
void Step2() {
  double sqrt3 = sqrt(3.0);
  random_device rd{};
  mt19937 gen{ rd() };
  normal_distribution<> normal(0, 1);
  cauchy_distribution<> cauchy(0, 1);
  poisson_distribution<> poisson(10.0);
  uniform_real_distribution<> uniform(-sqrt3, sqrt3);

  const int sizeOfSample[] = { 10, 100, 1000 }, size = 3;

  sample_t sample = NORMAL;
  //sample_t sample = CAUCHY;
  //sample_t sample = POISSON;
  //sample_t sample = UNIFORM

  vec x, medx, zR, zQ, ztr;
  vec data;

  if (sample == NORMAL) cout << "Normal:";
  else if (sample == CAUCHY) cout << "Cauchy:";
  else if (sample == POISSON) cout << "Poisson:";
  else cout << "Uniform:";
  cout << "\n";

  for (int i = 0; i < sizeOfSample[i]; ++i) {
    cout << "Size = " << sizeOfSample[i] << "\n";

    for (int k = 0; k < 1000; ++k) {

      for (int j = 0; j < sizeOfSample[i]; ++j) {
        if (sample == NORMAL) data.push_back(normal(gen));
        else if (sample == CAUCHY) data.push_back(cauchy(gen));
        else if (sample == POISSON) data.push_back(poisson(gen));
        else data.push_back(uniform(gen));
      }

      sort(data.begin(), data.end());

      x.push_back(FindX(data));
      medx.push_back(FindMedX(data));
      zR.push_back(FindzR(data));
      zQ.push_back(FindzQ(data));
      ztr.push_back(Findztr(data));

      data.clear();
    }// for (int k = 0; k < 1000; ++k)

    cout << "E(z) = " << FindX(x) << " " << FindX(medx) << " "\
      << FindX(zR) << " " << FindX(zQ) << " " << FindX(ztr) << "\n";
    cout << "D(z) = " << FindDX(x) << " " << FindDX(medx) << " "\
      << FindDX(zR) << " " << FindDX(zQ) << " " << FindDX(ztr) << "\n\n";

    x.clear();
    medx.clear();
    zR.clear();
    zQ.clear();
    ztr.clear();
  }

}

//Сгенерировать выборки размером 20 и 100 элементов.
//Построить для них боксплот Тьюки.
//Для каждого распределения определить долю выбросов экспериментально(сгенерировав выборку
//соответствующую распределению 1000 раз, и вычислив среднюю долю выбросов) 
//и сравнить с результатами, полученными теоретически.
void Step3() {
  double sqrt3 = sqrt(3.0);
  random_device rd{};
  mt19937 gen{ rd() };
  normal_distribution<> normal(0, 1);
  cauchy_distribution<> cauchy(0, 1);
  poisson_distribution<> poisson(10.0);
  uniform_real_distribution<> uniform(-sqrt3, sqrt3);

  const int sizeOfSample[] = { 20, 100 }, size = 2;

  //sample_t sample = NORMAL;
  sample_t sample = CAUCHY;
  //sample_t sample = POISSON;
  //sample_t sample = UNIFORM;

  //vec x, medx, zR, zQ, ztr;
  vec data;

  if (sample == NORMAL) cout << "Normal:";
  else if (sample == CAUCHY) cout << "Cauchy:";
  else if (sample == POISSON) cout << "Poisson:";
  else cout << "Uniform:";
  cout << "\n";

  for (int i = 0; i < sizeOfSample[i]; ++i) {
    cout << "Size = " << sizeOfSample[i] << "\n";

    //for (int k = 0; k < 1000; ++k) {

      for (int j = 0; j < sizeOfSample[i]; ++j) {
        
        if (sample == NORMAL) data.push_back(normal(gen));
        else if (sample == CAUCHY) data.push_back(cauchy(gen));
        else if (sample == POISSON) data.push_back(poisson(gen));
        else data.push_back(uniform(gen));
        
      }
      sort(data.begin(), data.end());
      for (int i = 0; i < data.size(); ++i)
        cout << data[i] << " ";
      cout << "\n";

      double Q1 = FindX14(data), Q3 = FindX34(data);
      double X1 = Q1 - 1.5 * (Q3 - Q1), X2 = Q3 + 1.5 * (Q3 - Q1);
      cout << "1st 3rd quartile:" << Q1 <<" "<< Q3 <<"\n";
      cout << "mustash:" << X1 << " " << X2 << "\n\n";

      data.clear();
    //}
  }


}


//4. Сгенерировать выборки размером 20, 60 и 100 элементов.
//Построить на них эмпирические функции распределения и ядерные
//оценки плотности распределения на отрезке[−4; 4] для непрерывных
//распределений и на отрезке[6; 14] для распределения Пуассона.

double f(vec& data, double hn, double x, double(*func)(double)) {
  double ans = 0.0;
  int n = data.size();
  for (int i = 0; i < n; ++i) 
    ans += func((x - data[i]) / hn);
  ans /= (n * hn);
  return ans;
}

void Step4() {
  double sqrt3 = sqrt(3.0);
  random_device rd{};
  mt19937 gen{ rd() };
  normal_distribution<> normal(0, 1);
  cauchy_distribution<> cauchy(0, 1);
  poisson_distribution<> poisson(10.0);
  uniform_real_distribution<> uniform(-sqrt3, sqrt3);

  const int sizeOfSample[] = { 20,60,100 }, size = 3;

  //sample_t sample = NORMAL;
  sample_t sample = CAUCHY;
  //sample_t sample = POISSON;
  //sample_t sample = UNIFORM;

  vec data;

  if (sample == NORMAL) cout << "Normal:";
  else if (sample == CAUCHY) cout << "Cauchy:";
  else if (sample == POISSON) cout << "Poisson:";
  else cout << "Uniform:";
  cout << "\n";

  for (int i = 0; i < size; ++i) {
    cout << "Size = " << sizeOfSample[i] << "\n";

    for (int j = 0; j < sizeOfSample[i]; ++j) {
      if (sample == NORMAL) data.push_back(normal(gen));
      else if (sample == CAUCHY) data.push_back(cauchy(gen));
      else if (sample == POISSON) data.push_back(poisson(gen));
      else data.push_back(uniform(gen));
    }

    //sort(data.begin(), data.end());

    //ядерные оценки
    double sigma = sqrt(FindDX(data));
    double hn = 1.05 * sigma * pow(data.size(), -0.2);
    double x, dx = 0.05;
    int iter = (int)(8.0 / dx), j=0;

    cout << "Kernel density:\n";
    hn /= 2;
    for (int iter1 = 1; iter1 <= 3; ++iter1, hn *= 2) {

      if (iter1 == 1) cout << "hn = hn/2\n";
      else if (iter1 == 2) cout << "hn = hn\n";
      else if (iter1 == 3) cout << "hn = hn*2\n";

      if (sample == POISSON) x = 6.0;
      else x = -4.0;

      while (j <= iter) {
        cout << f(data, hn, x, K1) << " ";
        ++j;
        x += dx;
      }
      j = 0;
      cout << "\n\n";
    }

    data.clear();
  }
    cout << "\n\n";
}


//Теоретическая вероятность выбросов
void Step5() {
  double sqrt3 = sqrt(3.0);
  random_device rd{};
  mt19937 gen{ rd() };
  normal_distribution<> normal(0, 1);
  cauchy_distribution<> cauchy(0, 1);
  poisson_distribution<> poisson(10.0);
  uniform_real_distribution<> uniform(-sqrt3, sqrt3);

  const int sizeOfSample[] = { 20,100 }, size = 2;

  //sample_t sample = NORMAL;
  //sample_t sample = CAUCHY;
  //sample_t sample = POISSON;
  sample_t sample = UNIFORM;

  vec data;

  if (sample == NORMAL) cout << "Normal:";
  else if (sample == CAUCHY) cout << "Cauchy:";
  else if (sample == POISSON) cout << "Poisson:";
  else cout << "Uniform:";
  cout << "\n";

  for (int i = 0; i < size; ++i) {
    cout << "Size = " << sizeOfSample[i] << "\n";

    for (int j = 0; j < sizeOfSample[i]; ++j) {
      if (sample == NORMAL) data.push_back(normal(gen));
      else if (sample == CAUCHY) data.push_back(cauchy(gen));
      else if (sample == POISSON) data.push_back(poisson(gen));
      else data.push_back(uniform(gen));
    }

    sort(data.begin(), data.end());

    for (int i = 0; i < data.size(); ++i)
      cout << data[i] << " ";

    cout << "\n";

    double Q1, Q3;
    if (sample == NORMAL) Q1 = -0.674, Q3 = 0.674;
    else if (sample == CAUCHY) Q1 = -1, Q3 = 1;
    else if (sample == POISSON) Q1 = 8, Q3 = 12;
    else Q1 = -0.866, Q3 = 0.866;
    double X1 = Q1 - 1.5 * (Q3 - Q1), X2 = Q3 + 1.5 * (Q3 - Q1);

    cout << " Q1 = " << Q1;
    cout << " Q3 = " << Q3 << "\n";
    cout << " X1 = " << X1;
    cout << " X2 = " << X2 << "\n";

    //доля выбросов
    int count = 0;
    for (int i = 0; i < data.size(); ++i) {
      if (data[i]<X1 || data[i]>X2)
        ++count;
    }
    cout << "outlier: " << ((double)count) / data.size();
    cout << "\n\n\n";

    data.clear();
  }
}


int main() {
  //Step1();
  //Step2();
  //Step3();
  //Step4();
  Step5();
  return 0;
}