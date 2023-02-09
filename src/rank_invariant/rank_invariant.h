// #include "../gudhi/Simplex_tree.h"
#include "../gudhi/Simplex_tree_multi.h"
#include "../utils/box.h"
#include "../utils/debug.h"
#include "../gudhi/Persistent_cohomology.h"

#include <omp.h>
#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include <tuple>
#include <iterator>  // for std::distance
#include <numeric>
#include <algorithm>




namespace Gudhi{

using Simplex_tree_std = Simplex_tree<Simplex_tree_options_full_featured>;
using Simplex_tree_multi = Simplex_tree<Simplex_tree_options_multidimensional_filtration>;
using value_type = Simplex_tree_options_multidimensional_filtration::Filtration_value::value_type;



template<typename T=int>
using Rectangle = std::tuple<std::vector<T>, std::vector<T>, int>;


inline std::vector<value_type> project_to_elbow(const std::vector<value_type> &x, value_type i, value_type j, value_type I, value_type J){
	utils::Box<value_type> top_left_zone({0,j+1}, {i-1,J-1});
	utils::Box<value_type> bottom_left_zone({1,0}, {i,j-1});
	utils::Box<value_type> right_zone({i+1,0}, {I-1,J-1});
	auto projection = x;
	if (bottom_left_zone.contains(x))
		projection[1] = j;
		// projection = {x[0], j};

	else if (top_left_zone.contains(x))
		projection[0] = i;
		// projection = {i,x[1]};

	else if (right_zone.contains(x))
		projection[1] = J-1;
		// projection = {x[0], J-1};
	
	// if (close_top && projection[1] == j){
	// 	projection[1] = j+0.1;
	// }
	return projection;
}

using Barcode = std::vector<std::pair<value_type, value_type>>;



inline Barcode compute_dgm(Simplex_tree_std &st, int degree){
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 2;
	int min_persistence = 0;
	Gudhi::persistent_cohomology::Persistent_cohomology<Gudhi::Simplex_tree<>, Gudhi::persistent_cohomology::Field_Zp> pcoh(st);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	auto persistent_pairs = pcoh.intervals_in_dimension(degree);
	return persistent_pairs;
}

using Elbow = std::vector<std::vector<unsigned int>>;
inline Elbow get_elbow(unsigned int i,unsigned int j,unsigned int I, unsigned int J){ 
	constexpr bool verbose = false;
	std::vector<std::vector<unsigned int>> out(I+J, std::vector<unsigned int>(2));
	if constexpr (verbose) std::cout << "Computing elbow " << i << " " << j << std::endl;
	unsigned int _i=0, _j=0;
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
rank_tensor get_2drank_invariant(const uintptr_t simplextree_ptr, const std::vector<unsigned int> &grid_shape, const int degree){
	constexpr bool verbose=false;
	Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	unsigned int I = grid_shape[0], J = grid_shape[1];
	rank_tensor out(I, std::vector<std::vector<std::vector<int>>>(
					J, std::vector<std::vector<int>>(
					I, std::vector<int>(
					J,0)))); // zero of good size
	// std::cout << I <<" " << J << std::endl;
	Simplex_tree_std _st;
	flatten((uintptr_t)(&st_multi), (uintptr_t)(&_st),0U); // copies the st_multi to a standard 1-pers simplextree
	
	#pragma omp parallel shared(out)
	{
		Simplex_tree_std st(_st); // copy for each core
		std::vector<int> vertices(st.dimension());

		#pragma omp for collapse(2)
		for (unsigned int i = 0 ; i < I; i++){
			for(unsigned int j = 0 ; j < J; j++){
				// Assign filtration values of the elbow
				// std::cout << i <<" " << j << std::endl;
				if constexpr (verbose) std::cout <<"\nElbow : "<<  i << " " << j << std::endl;
				for (auto &sh : st_multi.complex_simplex_range()){
					vertices.resize(0);
					for(const auto &vertex : st_multi.simplex_vertex_range(sh))
						vertices.push_back(vertex);
					const auto &multi_filtration = st_multi.filtration(sh);
					// auto int_multi_filtration = std::vector<int>(multi_filtration.begin(), multi_filtration.end());
					auto projection_to_elbow = project_to_elbow(multi_filtration,i ,j, I,J);
					// if constexpr(verbose) std::cout << "Projection : " << multi_filtration << " to " << projection_to_elbow << std::endl; 
					value_type elbow_filtration = std::reduce(projection_to_elbow.begin(), projection_to_elbow.end()); // sum
					// int elbow_filtration = projection_to_elbow[0] + projection_to_elbow[1];
					auto st_handle = st.find(vertices);
					st.assign_filtration(st_handle, elbow_filtration);
				}
				auto elbow = get_elbow(i,j,I,J);
				if constexpr(verbose) std::cout << "Computed elbow : " << elbow << std::endl;
				Barcode barcode = compute_dgm(st, degree);
				for(const auto &bar : barcode){
					unsigned int birth = bar.first;
					unsigned int death = bar.second == std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity() ? I+J-1: bar.second; // TODO FIXME 
					
					//Thresholds:
					if constexpr (verbose) std::cout <<"Bar " << birth << " " << death << std::endl;
					birth = std::max<unsigned int>(birth, j);
					death = std::min<unsigned int>(death, I + i);
					if constexpr (verbose) std::cout <<"Thresholded Bar " << birth << " " << death << std::endl;
					
					// only update rank of bars that are on the elbow // Does not work : top is open
					// if (birth > i+j || death < i+j)
					// 	continue;
	
					// for (int b = birth; b <= i+j; b++){
					// 	for (int d = i+j; d <= death; d++){
					for (unsigned int b = birth; b < death; b ++){
						for(unsigned int d = b; d < death; d++ ){
							const std::vector<unsigned int> &birth_coordinates = elbow[b];
							const std::vector<unsigned int> &death_coordinates = elbow[d];
							
							unsigned int b1 = birth_coordinates[0], b2 = birth_coordinates[1];
							unsigned int d1 = death_coordinates[0], d2 = death_coordinates[1];
							if ((b1 != d1 || b2 == j) && (b2 != d2 || d1 == i)){
								out[b1][b2][d1][d2]++;
								if constexpr (verbose) std::cout <<"Adding bar to rank : " << b << "/" <<birth << " " << d << "/" << death << " birth :"  << birth_coordinates << " death " << death_coordinates << std::endl;
							}
								
						}
					}
				}
			}
		}
	}
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
grid2d get_2Dhilbert(const uintptr_t simplextree_ptr, const std::vector<unsigned int> &grid_shape, const int degree){
	Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	unsigned int I = grid_shape[0], J = grid_shape[1];
	grid2d out(I, std::vector<int>(J,0)); // zero of good size
	// std::cout << I <<" " << J << std::endl;
	Simplex_tree_std _st;
	flatten((uintptr_t)(&st_multi), (uintptr_t)(&_st),0U); // copies the st_multi to a standard 1-pers simplextree
	#pragma omp parallel shared(out)
	{
		Simplex_tree_std st(_st); // copy for each core
		std::vector<int> vertices(st.dimension());

		#pragma omp for
		for (unsigned int height = 0 ; height < J; height++){
			// assigns simplices values 
			for (auto &sh : st_multi.complex_simplex_range()){
				vertices.resize(0);
				for(const auto &vertex : st_multi.simplex_vertex_range(sh))
					vertices.push_back(vertex);
				const auto &multi_filtration = st_multi.filtration(sh);
				value_type elbow_filtration = horizontal_line_filtration(multi_filtration, height); // sum
				auto st_handle = st.find(vertices);
				st.assign_filtration(st_handle, elbow_filtration);
			}
			Barcode barcode = compute_dgm(st, degree);
			for(const auto &bar : barcode){
				int birth = bar.first;
				int death = bar.second == std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity() ? I-1: bar.second; // TODO FIXME 
				for (int index = birth; index < death; index ++){
					out[index][height]++;
				}
			}
		}
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
// std::vector<std::vector<double>> sparsify_filtration(const uintptr_t newsplxptr, const std::vector<int> resolution){};

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
//     const uintptr_t splxptr, const uintptr_t newsplxptr, 
//     const std::vector<int> &elbow_coord,
//     const std::vector<std::vector<double>> &filtration_grid){ // not the best, maybe sorted sets
//     // Assumes that the filtrations values of splxptr are sparse; the grid is defined by
//     // num_parameters = 2 for the moment
//     // return;
//     Simplex_tree<option_std> &st = *(Gudhi::Simplex_tree<option_std>*)(newsplxptr);
// 	Simplex_tree<option_multi> &st_multi = *(Gudhi::Simplex_tree<option_multi>*)(splxptr);
//     unsigned int num_parameter = st_multi.get_number_of_parameters();
	
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
// 	Gudhi::get_2drank_invariant((uintptr_t)(&st),{10,10}, 0);
// 	return 0;
// }
