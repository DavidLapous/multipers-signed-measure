#pragma once

#include <omp.h>
#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include <tuple>
#include <iterator>  // for std::distance
#include <numeric>
#include <algorithm>
#include "Simplex_tree_multi.h"
#include "multi_filtrations/box.h"
//#include "temp/debug.h"
#include <gudhi/Persistent_cohomology.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>

namespace Gudhi::rank_invariant{

struct Simplex_tree_float { // smaller simplextrees
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef float Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false;
};

using multi_filtrations::Box;
using Simplex_tree_std = Simplex_tree<Simplex_tree_float>;
using Simplex_tree_multi = Simplex_tree<Simplex_tree_options_multidimensional_filtration>;
using value_type = Simplex_tree_options_multidimensional_filtration::Filtration_value::value_type;





template<typename T=int>
using Rectangle = std::tuple<std::vector<T>, std::vector<T>, int>;


inline void project_to_elbow(std::vector<value_type> &to_project, value_type i, value_type j, value_type I, value_type J){
	// Box<value_type> top_left_zone(0,j+1, i-1,J-1);
	// Box<value_type> bottom_left_zone(1,0, i,j-1);
	// Box<value_type> right_zone(i+1,0,I-1,J-1);
	Box<value_type> zone(1,0, i,j-1); // Bottom left zone
	auto &birth = zone.get_bottom_corner();
	auto &death = zone.get_upper_corner();
	if (zone.contains(to_project)){
		to_project[1] = j;
		// projection = {x[0], j};
		return;
	}
	birth[0] = 0; birth[1] = j+1; death[0] = i-1; death[1] = J-1;
	if (zone.contains(to_project)) //top left zone
		{to_project[0] = i; return;}
		// projection = {i,x[1]};
	birth[0] = i+1; birth[1] = 0; death[0] = I-1; //death[1] = J-1;
	if (zone.contains(to_project)) //right zone
		{to_project[1] = J-1; return;}
		// projection = {x[0], J-1};
	return;
	// // if (close_top && projection[1] == j){
	// // 	projection[1] = j+0.1;
	// // }
	// return to_project;
}

using Barcode = std::vector<std::pair<value_type, value_type>>;



inline Barcode compute_dgm(Simplex_tree_std &st, int degree){
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 2;
	constexpr int min_persistence = 0;
	Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree_std, Gudhi::persistent_cohomology::Field_Zp> pcoh(st);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	auto persistent_pairs = pcoh.intervals_in_dimension(degree);
	return persistent_pairs;
}

using Elbow = std::vector<std::vector<int>>;
inline Elbow get_elbow(int i,int j,int I, int J){ 
	constexpr bool verbose = false;
	std::vector<std::vector<int>> out(I+J, std::vector<int>(2));
	if constexpr (verbose) std::cout << "Computing elbow " << i << " " << j << std::endl;
	int _i=0, _j=0;
	while (_j < j){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_j++;
	}
	while(_i < i){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_i++;
	}
	while (_j < J){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_j++;
	}
	_j--;
	_i++;
	while(_i < I){
		out[_i+_j] = {_i,_j};
		if constexpr (verbose) std::cout << "    {" << _i << " " << _j << "}" << std::endl; 
		_i++;
	}
	out[I+J-1] = {I,J};
	return out;
}


// For 2_dimensional rank
using rank_tensor = std::vector<std::vector<std::vector<std::vector<int>>>>;
// assumes that the simplextree has grid coordinate filtration
rank_tensor get_2drank_invariant(const intptr_t simplextree_ptr, const std::vector<int> &grid_shape, const int degree){
	constexpr bool verbose=false;
	Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	int I = grid_shape[0], J = grid_shape[1];
	rank_tensor out(I, std::vector<std::vector<std::vector<int>>>(
					J, std::vector<std::vector<int>>(
					I, std::vector<int>(
					J,0)))); // zero of good size
	// std::cout << I <<" " << J << std::endl;
	Simplex_tree_std st_;
	flatten(st_, st_multi,0U); // copies the st_multi to a standard 1-pers simplextree

	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree;
	tbb::parallel_for(0, I,[&](int i){
		tbb::parallel_for(0,J, [&](int j){
			// gets the thread local variables
			Simplex_tree_std &st = thread_simplex_tree.local();
			const Elbow &elbow_container = get_elbow(i,j,I,J);
			if (st.num_simplices() == 0){ st = st_;}
			if constexpr (verbose) std::cout <<"\nElbow : "<<  i << " " << j << std::endl;

			Simplex_tree_multi::Filtration_value multi_filtration;
			auto sh_multi = st_multi.complex_simplex_range().begin(); // relies on the fact that this iterator is determinstic for two simplextrees having the same simplices
			auto sh_standard = st.complex_simplex_range().begin();
			auto _end = st.complex_simplex_range().end();
			for (; sh_standard != _end; ++sh_standard, ++sh_multi){
				multi_filtration = st_multi.filtration(*sh_multi);
				project_to_elbow(multi_filtration,i ,j, I,J);
				auto elbow_filtration = multi_filtration[0] + multi_filtration[1];
				st.assign_filtration(*sh_standard, elbow_filtration);
			}
			
			const Barcode barcode = compute_dgm(st, degree);
			for(const auto &bar : barcode){
				int birth = static_cast<int>(bar.first);
				int death = bar.second == std::numeric_limits<int>::infinity() ? I+J-1: static_cast<int>(bar.second); // TODO FIXME 
				
				//Thresholds:
				if constexpr (verbose) std::cout <<"Bar " << birth << " " << death << std::endl;
				birth = std::max<int>(birth, j);
				death = std::min<int>(death, I + i);
				if constexpr (verbose) std::cout <<"Thresholded Bar " << birth << " " << death << std::endl;

				for (int b = birth; b < death; b ++){
					for(int d = b; d < death; d++ ){
						const std::vector<int> &birth_coordinates = elbow_container[b];
						const std::vector<int> &death_coordinates = elbow_container[d];
						
						int b1 = birth_coordinates[0], b2 = birth_coordinates[1];
						int d1 = death_coordinates[0], d2 = death_coordinates[1];
						if ((b1 != d1 || b2 == j) && (b2 != d2 || d1 == i)){
							out[b1][b2][d1][d2]++;
						}
							
					}
				}
			}
			

		});
	});

	return out;
}



inline value_type horizontal_line_filtration(const std::vector<value_type> &x, value_type height){
	if (x[1] <= height)
		return x[0];
	else
		return std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity();
}

// inline assign_std_simplextree_from_multi(Simplex_tree_std& st,const Simplex_tree_multi& st_multi, function_type)






using grid2d = std::vector<std::vector<int>>;
grid2d get_2Dhilbert(const intptr_t simplextree_ptr, const std::vector<int> &grid_shape, const int degree){
	// Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	auto &st_multi = get_simplextree_from_pointer<options_multi>(simplextree_ptr);
	if (grid_shape.size() != 2){
		std::cerr << "Use a 2d grid shape."<<std::endl;
		return grid2d();
	}
	int I = grid_shape[0], J = grid_shape[1];
	grid2d out(I, std::vector<int>(J,0)); // zero of good size
	Simplex_tree_std _st;
	flatten(_st, st_multi,0U); // copies the st_multi to a standard 1-pers simplextree
	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree;
	tbb::parallel_for(0, I,[&](int height){
		Simplex_tree_std &st_std = thread_simplex_tree.local();
		if (st_std.num_simplices() == 0){ st_std = _st;}
		Simplex_tree_multi::Filtration_value multi_filtration;
		auto sh_standard = st_std.complex_simplex_range().begin();
		auto _end = st_std.complex_simplex_range().end();
		auto sh_multi = st_multi.complex_simplex_range().begin();
		for (;sh_standard != _end; ++sh_multi, ++sh_standard){
			multi_filtration = st_multi.filtration(*sh_multi);
			value_type horizontal_filtration = horizontal_line_filtration(multi_filtration, height);
			st_std.assign_filtration(*sh_standard, horizontal_filtration);
		}
		const Barcode barcode = compute_dgm(st_std, degree);
		for(const auto &bar : barcode){
			auto birth = bar.first;
			if (birth > I) // some birth can be infinite
				continue; 
			int death = bar.second > I ? I-1: bar.second; // TODO FIXME 
			for (int index = birth; index <= death; index ++){
				out[index][height]++;
			}
		}
	});
	return out;
	
}



using grid2d = std::vector<std::vector<int>>;
grid2d get_euler2d(const intptr_t simplextree_ptr, const std::vector<int> &grid_shape){
	// Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	auto &st_multi = get_simplextree_from_pointer<options_multi>(simplextree_ptr);
	if (grid_shape.size() != 2){
		std::cerr << "Use a 2d grid shape."<<std::endl;
		return grid2d();
	}
	int I = grid_shape[0], J = grid_shape[1];
	grid2d out(I, std::vector<int>(J,0)); // zero of good size
	for (auto &sh : st_multi.complex_simplex_range()){
		auto filtration = st_multi.filtration((sh));
		int sign = 1- 2*(st_multi.dimension(sh) % 2);
		for (int i=filtration[0]; i<I; i++)
			for (int j = filtration[1]; j<J; j++)
				out[i][j] += sign;
	}
	return out;
}





// std::vector<Rectangle<int>> signed_barcode(const rank_tensor &rank_invariant){
// 	Barcode barcode;
// 	compute_R_S_incl_excl(rank_invariant,barcode);
// 	std::vector<Rectangle<int>> out;
// 	out.reserve(barcode.size());
// 	for (const auto &bar_multiplicity : barcode){
// 		const Bar &bar = bar_multiplicity.first;
// 		const int &multiplicity = bar_multiplicity.second;
// 		Rectangle<int> rectangle({bar.first.first, bar.first.second}, {bar.second.first,bar.second.second}, multiplicity);
// 		out.push_back(rectangle);
// 	}
// 	return out;
// }

// TODO
// Turns the filtration values of the newsplxptr to integers, and returns the translation to get them back (up to resolution precision)
// std::vector<std::vector<double>> sparsify_filtration(const intptr_t newsplxptr, const std::vector<int> resolution){};

// template<typename T>
// class Elbow{
//     std::vector<std::vector<T>> grid;
//     int i, int j;

//     std::vector<T> push_forward(const std::vector<T> &point) const ;
//     Elbow(grid, i,j);
// };



// TODO
// Create a simplextree with filtration values being the one given by the simplextree multi projected on an elbow
// void flatten_elbow( //projects filtration to grid of filtration values
//     const intptr_t splxptr, const intptr_t newsplxptr, 
//     const std::vector<int> &elbow_coord,
//     const std::vector<std::vector<double>> &multi_filtration_grid){ // not the best, maybe sorted sets
//     // Assumes that the filtrations values of splxptr are sparse; the grid is defined by
//     // num_parameters = 2 for the moment
//     // return;
//     Simplex_tree<options_std> &st = *(Gudhi::Simplex_tree<options_std>*)(newsplxptr);
// 	Simplex_tree<options_multi> &st_multi = *(Gudhi::Simplex_tree<options_multi>*)(splxptr);
//     int num_parameter = st_multi.get_number_of_parameters();
	
//     std::vector<int> index(num_parameter);
//     for (const auto &simplex_handle : st_multi.complex_simplex_range()){
// 		// std::vector<int> simplex;
// 		// for (auto vertex : st_multi.simplex_vertex_range(simplex_handle))
// 		// 	simplex.push_back(vertex);
		
// 		// std::vector<double> f = st_multi.filtration(simplex_handle);
// 		// if (dimension <0)	 dimension = 0;
// 		// double new_filtration = l.push_forward(f)[dimension];
// 		// st.insert_simplex(simplex,new_filtration);
// 	} 
//     return;
// };

// TODO 
// Rank invariant from simplextree multi
// for elbow in grid, compute persistence, and extract rank invariant


// TODO 
// Rank invariant to signed measure


} // namespace Gudhi

// int main(){
// 	std::cerr << "0" << std::endl;
// 	Gudhi::Simplex_tree_multi st;
// 	std::cout << "1" << std::endl;
// 	st.insert_simplex_and_subfaces({0,1,2}, {0,0});
// 	std::cout << 3 << std::endl;
// 	Gudhi::get_2drank_invariant((intptr_t)(&st),{10,10}, 0);
// 	return 0;
// }
