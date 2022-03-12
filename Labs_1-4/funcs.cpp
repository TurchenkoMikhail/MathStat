#include <iostream>
#include <vector>

//first quartil
double FindX14(std::vector<double>& sortedData) {
  int n = sortedData.size();
  if (n % 4 == 0)
    return sortedData[(int)((n / 4) - 1)];
  else
    return sortedData[(int)(n / 4)];
}
//third quartil
double FindX34(std::vector<double>& sortedData) {
  int n = sortedData.size();
  if ((3 * n) % 4 == 0)
    return sortedData[(3 * n) / 4 - 1];
  else
    return sortedData[(int)((3 * n) / 4)];
}

double FindX(std::vector<double>& data) {
  double ans = 0.0;
  for (int i = 0; i < data.size(); ++i) 
    ans += data[i];
  return ans / data.size();
}

double FindDX(std::vector<double>& data) {
  int n = data.size();
  double X = FindX(data);
  double ans = 0.0;

  for (int i = 0; i < n; ++i) 
    ans += (data[i] - X) * (data[i] - X);
  return ans / n;
}

double FindMedX(std::vector<double>& sortedData) {
  int n = sortedData.size(), l;

  if (n % 2 == 0) {
    l = n / 2;
    return (sortedData[l - 1] + sortedData[l]) / 2.0;
  }
  else {
    l = (n - 1) / 2;
    return sortedData[l];
  }
}

double FindzR(std::vector<double>& sortedData) {
  return (sortedData[0] + sortedData[sortedData.size() - 1]) / 2.0;
}

double FindzQ(std::vector<double>& sortedData) {
  //zQ = (z(1/4) + z(3/4))/2
  int n = sortedData.size();
  double z14, z34;

  z14 = FindX14(sortedData);
  z34 = FindX34(sortedData);
  
  return (z14 + z34) / 2;
}

double Findztr(std::vector<double>& sortedData) {
  int n = sortedData.size();
  int r = (int)(n / 4);
  double ans = 0.0;
  for (int i = r; i < n - r; ++i)
    ans += sortedData[i];

  return ans / (n - 2 * r);
}