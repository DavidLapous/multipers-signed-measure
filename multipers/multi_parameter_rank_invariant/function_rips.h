#pragma once

#include "rank_invariant.h"

namespace Gudhi::rank_invariant::degree_rips{

using Simplex_tree_std = Simplex_tree<Simplex_tree_options_full_featured>; // TODO : fast persistence (rips is contigus, ...)
using Filtration_value = Simplex_tree_std::Filtration_value;
using signed_measure = std::pair< std::vector<std::vector<Filtration_value>>, std::vector<int> >  ;
using filtration_lists = std::vector<std::vector<Filtration_value>>;
using grid2d = std::vector<std::vector<int>>;
using mobius_inv_1 = std::pair<std::vector<Filtration_value>, std::vector<int>>;
using dissimilarity_type = std::pair< std::vector<Filtration_value>, std::vector<Filtration_value> >;
using semi_mobius_inversion = std::vector< dissimilarity_type>;


inline filtration_lists get_degree_filtrations(Simplex_tree_std& st){
	filtration_lists out(st.num_vertices());
	for (auto sh : st.skeleton_simplex_range(1)){
		if (st.dimension() == 0) continue;
		value_type filtration = st.filtration(sh);
		for (auto node : st.simplex_vertex_range(sh)){
			out[node].push_back(filtration);
		}
	}
	for (auto& filtrations : out){
		std::sort(filtrations.begin(), filtrations.end());
	}
	return out;
}

inline void dissimilarity_clean(std::vector<Filtration_value>& a, std::vector<Filtration_value>& b){
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    int i = 0; int j = 0;
    while(i<a.size() && j < b.size()){
        if(a[i] < b[j]) i++;
        else if(a[i] > b[j]) j++;
        else if(a[i] == b[j]){
            std::swap(a[i], a.back());
            a.pop_back();
            std::swap(b[j], b.back());
            b.pop_back();
        }
    }
}

void add_diff_to_signed_measure(const dissimilarity_type& previous,const dissimilarity_type& current, signed_measure& container, Filtration_value degree){
    auto& pts = container.first;
    auto& weights = container.second; 
    auto& a = previous.first;
    auto& b = previous.second;
    auto& c = current.first;
    auto& d = current.second;
    int i=0, j=0, k=0, l=0;
    // il faut parcourir a,b,c,d le long de la filtration, et ajouter / retirer ce qu'il manque
    std::vector<Filtration_value> temp(4);
    while(i < a.size() || j < b.size() || k < c.size() || l < d.size()){
        // f : first filtration
        // f = +old : add f as minus (if not in c)
        // f = -old : add f as pos (if not in d)
        // f = +new : add f in pos (as not in a)
        // f = -new : add f in neg (as not in d)
        Filtration_value f,g,h,t;
        f = i<a.size() ? a[i] : std::numeric_limits<Filtration_value>::infinity();
        g = j<b.size() ? b[j] : std::numeric_limits<Filtration_value>::infinity();
        h = k<c.size() ? c[k] : std::numeric_limits<Filtration_value>::infinity();
        t = l<d.size() ? d[l] : std::numeric_limits<Filtration_value>::infinity();
        temp = {f,g,h,t};
        unsigned int min_index = std::min_element(temp.begin(), temp.end()) - temp.begin(); // the first index of min filtration
        if (min_index == 0){
            if (f == h){ // f is guaranteed to be finite here
                i++;
                k++;
            }
            else{ //f and h are the same -> skip
                if(pts.back()[0] == f && pts.back()[1] == degree)
                    weights.back()--;
                else{
                    pts.push_back({f,degree});
                    weights.push_back(-1); 
                }
                i++; // we dealt with this f;
            }
        }
        else if (min_index == 1)
        {
            if(g == t){
                j++;l++;
            }
            else{
                if(pts.back()[0] == g && pts.back()[1] == degree)
                    weights.back()++;
                else{
                    pts.push_back({g,degree});
                    weights.push_back(1); 
                }
                j++; // we dealt with this g;
            }
        }
        else if (min_index == 2) { // we know here that h > f so we add ( there should be no duplicate in h t by dissimilarity_clean)
            if(pts.back()[0] == h && pts.back()[1] == degree)
                weights.back()++;
            else{
                pts.push_back({h,degree});
                weights.push_back(1); 
            }
            k++; // we dealt with this h;
        }
        else if (min_index == 3){
            if(pts.back()[0] == t && pts.back()[1] == degree)
                weights.back()--;
            else{
                pts.push_back({t,degree});
                weights.push_back(-1); 
            }
            l++; // we dealt with this h;
        }
    }
}

signed_measure mobius_inverse_axis_1(semi_mobius_inversion& dissimilarities_slices, int max_degree, float tolerance = 0){
    bool constexpr verbose = false;
    signed_measure out;
    int min_degree = max_degree - dissimilarities_slices.size();
    auto& pts = out.first;
    auto& weights = out.second; 
    // Adds first line as generator
    for (auto radius : dissimilarities_slices.back().first){ // degree d first
        if (pts.size() > 0 && pts.back()[0] == radius){
            weights.back()++;
        }
        else{
            pts.push_back({radius,static_cast<Filtration_value>(max_degree)});
            weights.push_back(1);
        }
    }
     for (auto radius : dissimilarities_slices.back().second){ // degree d first
        if (pts.size() > 0 && pts.back()[0] == radius){
            weights.back()--;
        }
        else{
            pts.push_back({radius,static_cast<Filtration_value>(max_degree)});
            weights.push_back(-1);
        }
    }
    return out;
    // Adds the rest of the lines while preserving mobius inversion structure
    for (int degree = max_degree-1; degree>=min_degree; degree--){
        //TODO : predict and compare // preciction is given by dissimilarities_slices[degree+1]
        // mu+ mu- vs nu+ nu- : compare +vs+ and -vs-
        add_diff_to_signed_measure(dissimilarities_slices[degree+1], dissimilarities_slices[degree], out, static_cast<Filtration_value>(degree));
    }
    return out;
}


using Barcode = std::vector<std::pair<Filtration_value, Filtration_value>>;
inline Barcode compute_dgm(Simplex_tree<Simplex_tree_options_full_featured> &st, int degree){
	st.initialize_filtration();
	constexpr int coeff_field_characteristic = 11;
	constexpr Filtration_value min_persistence = 0;
	bool persistence_dim_max = st.dimension() == degree;
	Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree_std, Gudhi::persistent_cohomology::Field_Zp> pcoh(st,persistence_dim_max);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	auto persistent_pairs = pcoh.intervals_in_dimension(degree);
	if constexpr (false) {
		std::cout << "Number of bars : " << persistent_pairs.size() << "\n";
	}
	return persistent_pairs;
}


// grid shape coord 1
signed_measure degree_rips_hilbert_signed_measure(Simplex_tree_std &st, int num_degrees, int homological_degree){
	constexpr bool verbose = false;
	if constexpr(verbose)
		tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);
    filtration_lists degree_filtration = get_degree_filtrations(st);
    
    // gets the maximum degree
    int max_degree = 0;
    for (const auto& node_degree_filtrations : degree_filtration)
        max_degree = std::max(max_degree, static_cast<int>(node_degree_filtrations.size()));
    int min_degree = std::max(max_degree-num_degrees, 0);

    // 
    semi_mobius_inversion dissimilarities_slices(max_degree-min_degree);
	tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree;
	tbb::parallel_for(min_degree, max_degree,[&](int degree){
		Simplex_tree_std &st_copy = thread_simplex_tree.local();
		if (st_copy.num_simplices() == 0){ st_copy = st;}
        if (degree>0){
            auto sh = st.skeleton_simplex_range(1).begin();
            auto sh_copy = st_copy.skeleton_simplex_range(1).begin();
            auto sh_end = st.skeleton_simplex_range(1).end();
            for (;sh != sh_end; ++sh, ++sh_copy){
                auto new_filtration = st.filtration(*sh);
                for (int node : st.simplex_vertex_range(*sh)){
                    if (degree_filtration[node].size() < degree){
                        new_filtration = std::numeric_limits<Filtration_value>::infinity();
                        break;
                    }
                    new_filtration = std::max(new_filtration, degree_filtration[node][max_degree - degree]);
                }
                st_copy.assign_filtration(*sh_copy, new_filtration);
                if constexpr (verbose){
                    Simplex_tree_multi::Filtration_value splx;
                    for (auto vertex : st.simplex_vertex_range(*sh))	splx.push_back(vertex);
                    std::cout << "Simplex " << splx << "/"<< st_copy.num_simplices() << " Filtration multi " << st.filtration(*sh) << " Filtration 1d " <<  st_copy.filtration(*sh_copy) << "\n";
                }
            }
        }

		const Barcode barcode = compute_dgm(st_copy, homological_degree);
        std::vector<Filtration_value> mu_plus;
        std::vector<Filtration_value> mu_minus;
        mu_plus.reserve(barcode.size());
        mu_minus.reserve(barcode.size());
		for(const auto &bar : barcode){
			auto birth = bar.first;
			auto death = bar.second;
			mu_plus.push_back(birth);
            mu_minus.push_back(death);
		}
        // dissimilarity_clean(mu_plus, mu_minus); // will be cleaned in the end
        dissimilarities_slices.push_back({mu_plus,mu_minus});
	});
	return mobius_inverse_axis_1(dissimilarities_slices, max_degree);
}


template<typename ... Args>
signed_measure degree_rips_hilbert_signed_measure(const intptr_t simplextree_ptr, Args...args){
	auto &st = get_simplextree_from_pointer<Simplex_tree_options_full_featured>(simplextree_ptr);
	return degree_rips_hilbert_signed_measure(st, args...);
}

}