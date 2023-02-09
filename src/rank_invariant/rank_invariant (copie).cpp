#include "../gudhi/Simplex_tree.h"
#include "../gudhi/Simplex_tree_multi.h"
#include "../mma_cpp/box.h"

#include "../gudhi/Persistent_cohomology.h"

#include <omp.h>
#include <iostream>
#include <vector>
#include <utility>  // std::pair
#include <tuple>
#include <iterator>  // for std::distance

// #include "rank_decomposition.cpp"



namespace Gudhi{

// TODO
// std::vector<std::vector<double>> get_filtration_values(const uintptr_t splxptr){
//     Simplex_tree<option_multi> &st = *(Gudhi::Simplex_tree<option_multi>*)(splxptr);
//     std::vector<std::vector<value_type>> filtration_values(st.get_number_of_parameters());

//     for (const auto &simplex_handle : st.complex_simplex_range()){
		
		
// 	}
// }

// template<typename T>
// struct Rectangle{
//     std::vector<T> birth;
//     std::vector<T> death;
//     int multiplicity;
// };
using Simplex_tree_std = Simplex_tree<Simplex_tree_options_full_featured>;
using Simplex_tree_multi = Simplex_tree<Simplex_tree_options_multidimensional_filtration>;
using value_type = Simplex_tree_options_multidimensional_filtration::Filtration_value::value_type;
// using Vineyard::value_type;

// using Vertex_handle = Simplex_tree_std::Vertex_handle;
// using typeVectorVertex = std::vector<Vertex_handle>;

// struct simplex_node{
// 	double grid_x;
// 	double grid_y;
// 	typeVectorVertex face;
// };



// rank rank_invariant(const uintptr_t simplextree_ptr, const int degree = 1, const std::vector<int> &resolution = {10,10}){
//     Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
//     std::vector<simplex_node> simplices; 
//     std::vector<float> x_values;
//     std::vector<float> y_values;
//     rank rank_inv;

//     // for (auto &SimplexHandle : st_multi.complex_simplex_range()){
//     //     const auto filtration = st_multi.filtration(SimplexHandle);
//     //     if (filtration.size() != 2) return rank_inv;
//     //     std::vector<int> vertices;
//     //     for(const auto &vertex : st_multi.simplex_vertex_range(SimplexHandle))    vertices.push_back(vertex);
//     //     simplex_node splx;
//     //     splx.face.swap(vertices);
//     //     splx.grid_x = filtration[0], splx.grid_y = filtration[1];
//     //     x_values.push_back(filtration[0]);
//     //     y_values.push_back(filtration[1]);
//     //     simplices.push_back(splx);
//     // }
//     // int x_range = resolution[0];
//     // int y_range = resolution[1];
//     // std::vector<simplex_node> simplices_integer=simplices; // book-keeping: keep original simplex grades
//     // discretize_filtration(simplices_integer, x_values, y_values, x_range, y_range);
	
	
//     // zero_rank(rank_inv,x_range+1,y_range+1);
//     // compute_rank_invariant(simplices_integer, x_range, y_range, degree, rank_inv);
//     // return rank_inv;
//     // barcode b;
//     // compute_R_S_incl_excl(rank_inv,b);
//     // return b;
// }



template<typename T=int>
using Rectangle = std::tuple<std::vector<T>, std::vector<T>, int>;


int get_elbow_filtration(const std::vector<int> &x, int i, int j, int I, int J){
	Vineyard::Box<int> bottom_left_zone({0,0}, {i,j});
	Vineyard::Box<int> bottom_right_zone({i+1,0}, {I,j});
	Vineyard::Box<int> top_zone({0,j+1}, {I,J});
	if (bottom_left_zone.contains(x))
		return i+x[1]; // return {i, x[1]};
	else if (bottom_right_zone.contains(x))
		return x[0]+ j; // return {x[0], j};
	else 
		return I+j;// return {I+1,J+1};

}


using Barcode = std::vector<std::pair<value_type, value_type>>;



Barcode compute_dgm(Simplex_tree_std &st, int degree){
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 2;
	int min_persistence = 0;
	Gudhi::persistent_cohomology::Persistent_cohomology<Gudhi::Simplex_tree<>, Gudhi::persistent_cohomology::Field_Zp> pcoh(st);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	auto persistent_pairs = pcoh.intervals_in_dimension(degree);
	return persistent_pairs;
}

using Elbow = std::vector<std::vector<int>>;
Elbow get_elbow(int i, int j,int I,int J){
	std::vector<std::vector<int>> out(I+J+2, std::vector<int>(2));
	int _i=0, _j=0;
	while (_i < i){
		out[_i+_j] = {_i,_j};
		_i++;
	}
	while(_j < j){
		out[_i+_j] = {_i,_j};
		_j++;
	}
	while (_i < I){
		out[_i+_j] = {_i,_j};
		_i++;
	}
	while(_j < J){
		out[_i+_j] = {_i,_j};
		_j++;
	}
	return out;
}


// For 2_dimensional rank
using rank_tensor = std::vector<std::vector<std::vector<std::vector<int>>>>;
// assumes that the simplextree has grid coordinate filtration
rank_tensor get_2drank_invariant(const uintptr_t simplextree_ptr, const std::vector<unsigned int> &grid_shape, const int degree = 1){
	Simplex_tree_multi &st_multi = *(Simplex_tree_multi*)(simplextree_ptr);
	unsigned int I = grid_shape[0], J = grid_shape[1];
	rank_tensor out(I, std::vector<std::vector<std::vector<int>>>(
					J, std::vector<std::vector<int>>(
					I, std::vector<int>(
					J,0)))); // zero of good size
	// std::cout << I <<" " << J << std::endl;
	Simplex_tree_std _st;
	flatten((uintptr_t)(&st_multi), (uintptr_t)(&_st),0U); // copies the st_multi to a standard 1-pers simplextree
	
	// #pragma omp parallel shared(out) 
	{
		Simplex_tree_std st(_st); // copy for each core

		// #pragma omp for collapse(2)
		for (unsigned int i = 0 ; i < I; i++){
			for(unsigned int j = 0 ; j < J; j++){
				// Assign filtration values of the elbow
				// std::cout << i <<" " << j << std::endl;
				for (auto &sh : st_multi.complex_simplex_range()){
					std::vector<int> vertices;
					for(const auto &vertex : st_multi.simplex_vertex_range(sh))
						vertices.push_back(vertex);
					auto _filtration = st_multi.filtration(sh);
					auto filtration = std::vector<int>(_filtration.begin(), _filtration.end());
					value_type elbow_filtration = get_elbow_filtration(filtration, i,j,I,J);
					auto st_handle = st.find(vertices);
					st.assign_filtration(st_handle, elbow_filtration);
				}
				auto elbow = get_elbow(i,j,I,J);
				Barcode barcode = compute_dgm(st, degree);
				for(auto bar : barcode){
					int birth = bar.first;
					int death = bar.second == std::numeric_limits<Simplex_tree_std::Filtration_value>::infinity() ? I+J+1 : bar.second;
					std::cerr << birth << " " << death << std::endl;
					std::vector<int> birth_coordinates = elbow[birth];
					std::vector<int> death_coordinates = elbow[death];
					std::cerr << birth_coordinates[0] << " " << birth_coordinates[1] << " " << death_coordinates[0] <<  " " <<death_coordinates[1] << std::endl;
					// out[birth_coordinates[0]][birth_coordinates[1]][death_coordinates[0]][death_coordinates[1]]++;
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

int main(){
	std::cerr << "0" << std::endl;
	Gudhi::Simplex_tree_multi st;
	std::cout << "1" << std::endl;
	st.insert_simplex_and_subfaces({0,1,2}, {0,0});
	std::cout << 3 << std::endl;
	Gudhi::get_2drank_invariant((uintptr_t)(&st),{10,10}, 0);
	return 0;
}
