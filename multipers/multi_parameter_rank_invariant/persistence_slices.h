#include "gudhi/Persistent_cohomology.h"
#include "gudhi/Simplex_tree_multi.h"


namespace Gudhi::multiparameter{


struct Simplex_tree_float { // smaller simplextrees
  typedef linear_indexing_tag Indexing_tag;
  typedef int Vertex_handle;
  typedef float Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false; // TODO OPTIMIZATION : maybe make the simplextree contiguous when calling grid_squeeze ?
  static const bool link_nodes_by_label = true;
  static const bool stable_simplex_handles = false;
};
// using Simplex_tree_float = Simplex_tree_options_fast_persistence;

using multi_filtrations::Box;
using Simplex_tree_std = Simplex_tree<Simplex_tree_float>;
using Simplex_tree_multi = Simplex_tree<Simplex_tree_options_multidimensional_filtration>;
using value_type = Simplex_tree_options_multidimensional_filtration::Filtration_value::value_type;


using Barcode = std::vector<std::pair<Simplex_tree_std::Filtration_value, Simplex_tree_std::Filtration_value>>;
inline Barcode compute_dgm(Simplex_tree_std &st, int degree){
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 11;
	constexpr Simplex_tree_std::Filtration_value min_persistence = 0;
	bool persistence_dim_max = st.dimension() == degree;
	Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree_std, Gudhi::persistent_cohomology::Field_Zp> pcoh(st,persistence_dim_max);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	const auto& persistent_pairs = pcoh.intervals_in_dimension(degree);
	if constexpr (false) {
		std::cout << "Number of bars : " << persistent_pairs.size() << "\n";
	}
	return persistent_pairs;
}


template <typename degree_type>
inline std::vector<Barcode> compute_dgms(Simplex_tree_std &st, std::vector<degree_type> degrees){
	std::vector<Barcode> out(degrees.size());
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 11;
	constexpr Simplex_tree_std::Filtration_value min_persistence = 0;

	bool persistence_dim_max=false;
	for (auto degree : degrees){
		if (st.dimension() == degree) {
			persistence_dim_max = true;
			break;
		}
	} 
	Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree_std, Gudhi::persistent_cohomology::Field_Zp> pcoh(st,persistence_dim_max);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	for (auto i =0u; i<degrees.size();i++){
		out[i] = pcoh.intervals_in_dimension(degrees[i]);
	}
	return out;
}

}