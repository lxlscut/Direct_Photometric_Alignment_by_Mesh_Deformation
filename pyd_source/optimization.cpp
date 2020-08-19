// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <vl/generic.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;


pybind11::array_t<uint> optimize(pybind11::array_t<int> &triangle,pybind11::array_t<float> &triangle_cofficient,
	pybind11::array_t<float> &weight,pybind11::array_t<float> &location,pybind11::array_t<float> &b,pybind11::array_t<int> &vertices,
	pybind11::array_t<float> &cofficients_g,pybind11::array_t<float> &bbss,float &lambda)
{
	auto r1 = triangle.unchecked<3>();
	auto r2 = triangle_cofficient.unchecked<2>();
	auto r3 = weight.unchecked<3>();
	auto r4 = location.unchecked<2>();
	auto r5 = b.unchecked<1>();
	auto r6 = vertices.unchecked<3>();
	auto r8 = cofficients_g.unchecked<3>();
	auto r9 = bbss.unchecked<2>();

	pybind11::array_t<float> result = pybind11::array_t<float>(vertices.shape()[0] * vertices.shape()[1] * 2);
	auto r7 = result.mutable_unchecked<1>();
	//create a sparse matrix
	vector<Eigen::Triplet<float>> m;

	int width = vertices.shape()[1];
	cout << "width is :" << width << " the height is:" << vertices.shape()[0] << endl;
	//the total constrains is triangle numbers add weight numbers
	int constrain_num = triangle.shape()[0] + weight.shape()[0];
	//reserve memory
	m.reserve(constrain_num);
	//insert the constrain value
	int index = 0;
	//1.insert the optical point constrains
	for (int j=0;j<weight.shape()[0];j++)
	{
		for (int dim = 0; dim < 2; dim++)
		{
			if(j<10)
			{
				cout << "the row is:" << r4(j, 0) << "the col is" << r4(j, 1) << endl;
			}
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1))*2+dim, r3(j, 0, dim));
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1)+1)*2+dim, r3(j, 1, dim));
			m.emplace_back(index, ((r4(j, 0)+1)*width + r4(j, 1)+1)*2+dim, r3(j, 2, dim));
			m.emplace_back(index, ((r4(j, 0)+1)*width + r4(j, 1))*2+dim, r3(j, 3, dim));
		}
		index++;
	}
	cout << "the optical constrains inserted done..." << endl;
	//2.insert the gradient constrains
	for (int j = 0; j < weight.shape()[0]; j++)
	{
		for (int dim = 0; dim < 2; dim++)
		{
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1)) * 2 + dim, r8(j, 0, dim));
			m.emplace_back(index, (r4(j, 0)*width + r4(j, 1) + 1) * 2 + dim, r8(j, 1, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1) + 1) * 2 + dim, r8(j, 2, dim));
			m.emplace_back(index, ((r4(j, 0) + 1)*width + r4(j, 1)) * 2 + dim, r8(j, 3, dim));
			index++;
		}
	}
	cout << "insert the gradient constrains done ..." << endl;
	//3.insert the triangle constrains
	for (int t = 0; t < triangle.shape()[0]; t++) {
		for (int dim = 0; dim < 2; dim++) {
			float u = r2(t, 0);
			float v = r2(t, 1);
			int vertice_a = r1(t, 0, 0) * width + r1(t, 0, 1);
			int vertice_b = r1(t, 1, 0) * width + r1(t, 1, 1);
			int vertice_c = r1(t, 2, 0) * width + r1(t, 2, 1);
			if (dim == 0) {
				m.emplace_back(index, vertice_a * 2, lambda * 1);
				m.emplace_back(index, vertice_b * 2, lambda * (u - 1));
				m.emplace_back(index, vertice_c * 2, lambda * (-u));
				m.emplace_back(index, vertice_c * 2 + 1, lambda * (-v));
				m.emplace_back(index, vertice_b * 2 + 1, lambda * (v));
				index++;
			}
			else
			{
				m.emplace_back(index, vertice_a * 2 + 1, lambda);
				m.emplace_back(index, vertice_b * 2 + 1, lambda * (u - 1));
				m.emplace_back(index, vertice_c * 2 + 1, lambda * (-u));
				m.emplace_back(index, vertice_b * 2, lambda * (-v));
				m.emplace_back(index, vertice_c * 2, lambda * (v));
				index++;
			}
		}
	}
	cout << "insert the triangle constrains done ..." << endl;
	//set the value of b
	VectorXd bb = VectorXd::Zero(triangle.shape()[0] * 2 + weight.shape()[0]*3);
	VectorXd x = VectorXd::Zero(vertices.shape()[0] * vertices.shape()[1] * 2);
	for (int i=0;i<b.shape()[0];i++)
	{
		bb[i] = r5(i);
	}
	for (int j= 0;j<bbss.shape()[0];j++)
	{
		bb[b.shape()[0] + j * 2] = r9(j,0);
		bb[b.shape()[0] + j * 2 + 1] = r9(j,1);
	}
	cout << "set the b value done ..." << endl;
	//VectorXd bb = VectorXd::Zero(triangle.shape()[0] * 2);
	//solve the equation
	LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
	SparseMatrix<double> A(index, vertices.shape()[0] * vertices.shape()[1] * 2);
	A.setFromTriplets(m.begin(), m.end());
	lscg.compute(A);
	x = lscg.solve(bb);
	for (int i=0;i<vertices.shape()[0]*vertices.shape()[1]*2;i++)
	{
		r7(i) = x[i];
	}
	return result;
}

PYBIND11_MODULE(optimization,m)
{
	m.doc()= "the optimization of vertices of the image" ;
	m.def("optimize",&optimize);
}


