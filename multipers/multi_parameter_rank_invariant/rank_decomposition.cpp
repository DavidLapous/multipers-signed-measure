/** 
						 Rank decompositions of 2-parameter filtrations over a grid 
													Code by Jingyi Li and Steve Oudot
												 Copyright 2022, all rights reserved
**/


#include <gudhi/graph_simplicial_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#include <iostream>
#include <utility> 
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <omp.h>

using Simplex_tree = Gudhi::Simplex_tree<>;
using Vertex_handle = Simplex_tree::Vertex_handle;
using Simplex_handle =  Simplex_tree::Simplex_handle;
using Filtration_value = Simplex_tree::Filtration_value;
using typeVectorVertex = std::vector<Vertex_handle>;
using typePairSimplexBool = std::pair<Simplex_tree::Simplex_handle, bool>;
using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree, Field_Zp >;

typedef std::vector<std::vector<std::vector<std::vector<int> > > > rank; 
typedef std::pair<int,int> entry;
typedef std::pair<entry,entry> bar;
typedef std::map<bar, int> barcode;
typedef std::map<Simplex_handle,entry> simplex_grid_map;

using std::cout; 
using std::cerr;
using std::endl; 
using std::string;
using std::ifstream; 
using std::vector;

struct simplex_node{
	float grid_x;
	float grid_y;
	typeVectorVertex face;
};


constexpr bool verbose = false;




bool read_input_filtration(string filename, int rank_dim, vector<simplex_node>& simplices,
				 vector<float>& x_values, vector<float>& y_values) {

	// Variables for reading the file
	int line_i = 1;
	string line;
	vector<string> block_lines;
	int vertice_id = 0;
	vector<int> rank_num;

	// open file (read-only mode)
	ifstream input_file(filename);
	if (!input_file.is_open()) {
		cerr << "Could not open the file - '"
	 << filename << "'" << endl;
		return false;
	}

	// read header
	getline(input_file, line);
	assert(line == "scc2020");
		
	//store remaining lines in block_lines, and numbers of simplices of each dimension in rank_num
	while (getline(input_file, line)){
		if(line[0]!='#'){

			if(line_i<2){ // num parameters (assumed to be 2)
				line_i+=1;
				continue;
			}
			else if(line_i==2){ // block sizes
				std::stringstream ss(line);
				int out;
				while(ss>>out){
					rank_num.push_back(out);
				}
				// rank_num.pop_back(); //WTF?
				line_i += 1;
			}
			else{
				block_lines.push_back(line);
				line_i += 1;
			}
		}
		
	}
	input_file.close();

	// check that the filtration has simplice of dimension up to rank_dim+1
	if (rank_num.size() < rank_dim+2) {
		std::cout << "ERROR: homology dimension (" << rank_dim
				<< ") too large for simplicial complex dimension ("
				<< rank_num.size()-1 << ")" << std::endl;
		return false;
	}

		
	// extract simplices from input lines
	int shift = 0;
	int simplices_len=block_lines.size(); // number of chains 
	for(int i=rank_num.size()-1; i>=0; --i){ //for block
		// for(int j=simplices_len-shift-rank_num[i]; j<simplices_len-shift; ++j){ // for chain in block
		for (int j = 0; j< rank_num[i]; j++){
			std::stringstream ss(block_lines[block_lines.size()-1 - (j + shift)]);
			float grid_x, grid_y;
			int face1,face2,face3,face4;
			simplex_node simplex;
			int count=0;
			string out;
			while(ss>>out){
				if(count==0){
					grid_x = std::stof(out);
					++count;
				}else if(count==1){
					grid_y = std::stof(out);
					++count;
				}else if(count==2){
					++count;
					continue;
				}else{
					simplex.face.push_back(std::stoi(out));
				}
			}
			if(simplex.face.size()==0){
				simplex.face={vertice_id};
				vertice_id += 1;
			}
			
			if(simplex.face.size() > rank_dim+2) 
				continue;  // disregard simplices of dimension > rank_dim+1
			
			simplex.grid_x = grid_x;
			simplex.grid_y = grid_y;
			x_values.push_back(grid_x);
			y_values.push_back(grid_y);
			simplices.push_back(simplex);
			
		}
		shift += rank_num[i];
	}
	// cerr << shift << endl;

	// crop rank-num to consider only simplices of dimension up to rank_dim+1
	vector<int> rank_num2;
	for (int i=0; i<rank_num.size(); i++)
		if (i>=rank_num.size()-rank_dim-2)
			rank_num2.push_back(rank_num[i]);
	// rank_num = rank_num2;
	rank_num.swap(rank_num2);
		
	// convert simplex representation from array of facet indices to array of vertex indices
	shift = rank_num[rank_num.size()-1] + rank_num[rank_num.size()-2]; // size of 1 skeleton
	int shift_of_boundaries;
	for(int i=rank_num.size()-3; i>=0; --i){ // starts from 2 simplices
		shift_of_boundaries = shift - rank_num[i+1];
		// for(int j=shift; j<(shift+rank_num[i]); ++j){
		for (int j = 0; j< rank_num[i]; j++){
			simplex_node simplex = simplices[j+shift];
			// cerr <<"LINE : ";
			// for (auto vertex : simplex.face) cerr << vertex << " ";
			// cerr <<"\n";
			typeVectorVertex face = simplex.face;
			typeVectorVertex new_face;
			for(int k=0; k<face.size(); ++k){
				typeVectorVertex vertices = simplices[shift_of_boundaries+face[k]].face;
				for(int l=0; l<vertices.size(); ++l){
					int vertex = vertices[l];
					if (std::find(new_face.begin(), new_face.end(), vertex) == new_face.end()){
						new_face.push_back(vertex);
					}
				}		
			}
			simplices[j+shift].face=new_face;
			// cerr <<"NEW8LINE : ";
			// for (auto vertex : new_face) cerr << vertex << " ";
			// cerr <<"\n";

		}
		shift += rank_num[i];
	}

	std::cout << "simplicial complex size: " << simplices.size() << std::endl;

	return true;
}



// convert the float grid coord to integer grid coord and scale it according to grid size
void discretize_filtration(vector<simplex_node>& simplices,
				 vector<float>& x_values, vector<float>& y_values,
				 int x_range, int y_range) {

	float x_min = *min_element(x_values.begin(),x_values.end()); 
	float y_min = *min_element(y_values.begin(),y_values.end()); 
	float x_max = *max_element(x_values.begin(),x_values.end()); 
	float y_max = *max_element(y_values.begin(),y_values.end()); 
	float x_interval = (x_max-x_min)/x_range;
	float y_interval = (y_max-y_min)/y_range;


	for(int i=0; i<simplices.size(); ++i){
		float grid_x = simplices[i].grid_x;
		float grid_y = simplices[i].grid_y;

		if(x_interval==0)
			simplices[i].grid_x=0;
		else
			simplices[i].grid_x = round((grid_x-x_min)/x_interval);

		if(y_interval==0)
			simplices[i].grid_y=0;
		else
			simplices[i].grid_y = round((grid_y-y_min)/y_interval);

		// std::cout << "Simplex grade: (" << simplices_integer[i].grid_x
		// 	  << "," << simplices_integer[i].grid_y << ")\n";

	}
}



vector<entry> create_stair(entry elbow, entry end_point){
	vector<entry> stair;
	int i,j,k,l;
	if constexpr (verbose) std:: cout << "\n COMPUTING ELBOW :\n"; 
	if(elbow.first==0 || elbow.second==end_point.second){

		for(i=0; i<=end_point.second; ++i){
						stair.push_back(entry(0,i));
						if constexpr (verbose)std::cout << 0 << i << std::endl;
		}
		--i;
		for(j=1; j<=end_point.first; ++j){
						stair.push_back(entry(j,i));
						if constexpr (verbose)std::cout << j<<i<< std::endl;
		}
	}
		else{
			for(i=0; i<=elbow.second; ++i){
						stair.push_back(entry(0,i));
						if constexpr (verbose)std::cout << 0 << i << std::endl;
		}
		--i;
		for(j=0; j<elbow.first; ++j){
						stair.push_back(entry(j+1,i));
						if constexpr (verbose)std::cout << j+1<<i << std::endl;
		}
		--j;
		for(k=i+1; k<=end_point.second; ++k){
						stair.push_back(entry(j+1,k));
						if constexpr (verbose)std::cout << j+1 << k << std::endl;
		}
		--k;
		for(l=j+2; l<=end_point.first; ++l){
						stair.push_back(entry(l,k));
						if constexpr (verbose)std::cout << l<<k<< std::endl;
		}
		}
		return stair;
}


entry proj_grid_to_path(entry simplex_node, entry elbow, int x_range,int y_range){
// input: elbow = [i,j]
// output: stair = [0,0] ->...->[i,0] ->...->[i,j]->...->[x_range,j]->...->[x_range,y_range]
	int x = simplex_node.first;
	int y = simplex_node.second;
	int x0 = elbow.first;
		int y0 = elbow.second;
		entry proj_p;

		if((x==0 && y<=y0) || (y==y0 && x<=x0) || (y>=y0 && x==x0) || (y==y_range && x>=x0)){
				proj_p.first = x;
				proj_p.second = y;
		}
		else if(x<=x0 and y<=y0){ // bottom left
				proj_p.first = x;
				proj_p.second = y0;
		}
		else if(x<=x0 and y>=y0){ //top left
				proj_p.first = x0;
				proj_p.second = y;
		}
		else{
				proj_p.first = x;
				proj_p.second = y_range;
		}
	return proj_p;
}

// the following assumes the rank invariant is initially filled in with zeros
// it makes sure each entry in the rank invariant's table is filled in by exactly one path,
// hence only one thread
void compute_rank_invariant (vector<simplex_node>& simplices, int x_range, int y_range,
					 int rank_dim, rank& rank_inv) {
		
		entry end_point(x_range,y_range);

#pragma omp parallel // num_threads(8)
	{
	
		//initialize a simplex tree
		Simplex_tree st;
		
		for(int i=0; i<simplices.size();++i){
			auto &simplex = simplices[i].face; 
			assert (simplex.size()-1 <= rank_dim+2); // WARNING : This assumes that the scc is a simplical complex
			std::pair<Simplex_handle, bool> insert_return = st.insert_simplex(simplex, Filtration_value(i));//initialize the simplex tree
			
			if(insert_return.second==false)	{
				cerr << "ERROR:insert fail!!! with simplex "; //}
				for( auto vertex : simplex) cerr << vertex << " ";
				cerr << endl; 
			}
			
		}

		// Build the map from Simplex handle to grid node coordinates (x,y)
		simplex_grid_map sg_map;
		int i=0;
		for (auto f_simplex : st.filtration_simplex_range()) {
			sg_map[f_simplex] = entry(simplices[i].grid_x,simplices[i].grid_y);
			++i;
		}


		// Loop over the elbows (in parallel)
#pragma omp for
		for(int i=0; i<=x_range; ++i){
			for(int j=0; j<=y_range; ++j){
				if constexpr(verbose) cout << " \r \r elbow  : " << i << " " << j<< "          "<< std::endl;
		
	entry elbow(i,j);

	// create corresponding stair path in the grid
	vector<entry> stair = create_stair(elbow, end_point);

	//update the filtration values of the simplices in the simplex tree
	for(Simplex_handle sh: st.complex_simplex_range()){
		entry node_grid = sg_map[sh];
		entry proj_p = proj_grid_to_path(node_grid, elbow, x_range, y_range);
		st.assign_filtration(sh,proj_p.first+proj_p.second);
	}

	//re-order the simplices by filtration values
	st.initialize_filtration();

	// compute persistence (field characteristic = 11)
	constexpr int coeff_field_characteristic = 2;
	int min_persistence = 0;
	Persistent_cohomology pcoh(st);
	pcoh.init_coefficients(coeff_field_characteristic);
	pcoh.compute_persistent_cohomology(min_persistence);
	auto persistent_pairs = pcoh.intervals_in_dimension(rank_dim);


	//extract rank invariant along path from 1-d barcode
	for (auto pair : persistent_pairs) {                
		// update rank invariant only between (i,0) and (x_range,j) along the stair
		int barcode_sid = std::max((int)pair.first, j);
		if (pair.second>x_range+y_range) // infinite endpoint
			pair.second = x_range+y_range+1;
		int barcode_eid = std::min((int)pair.second, y_range+i+1);
		if constexpr (verbose) std::cout <<"BAR : "<<  pair.first << ":"<< barcode_sid << " / " << pair.second<< ":"<< barcode_eid<< std::endl;
		for(int r_s=barcode_sid; r_s<barcode_eid; ++r_s){
			for(int r_e=r_s; r_e<barcode_eid; ++r_e){
				int ii = stair[r_s].first;
				int jj = stair[r_s].second;
				int kk = stair[r_e].first;
				int ll = stair[r_e].second;

				// update rank only once if (ii,jj) and (kk,ll) share a coordinate
				if ((ii!=kk || jj==j) && (jj!=ll || kk==i)){
					rank_inv[ii][jj][kk][ll]++;
					if constexpr (verbose) std::cout <<"Adding bar : "<<  ii<< " " << jj<< " " <<  kk<< " " << ll<< std::endl; 
				}
			}
		}   
	}	  
			}
		}
	}
}


void zero_rank (rank& r, int n, int m) {  // n columns, m rows in the grid (column-dominant)
		r.resize(n);
		for (vector<vector<vector<int> > >& u: r){
				u.resize(m);
				for (vector<vector<int> >& v: u){
						v.resize(n);
						for (vector<int>& w: v) {
								w.resize(m);
								w.assign(m, 0);
						}
				}
		}
}



int multp (const rank& r, int i, int j, int k, int l) {
//The formula is m(s,t+) = r(s,t) − r((s.x −1,s.y),t) − r((s.x,s.y −1),t) + r((s.x −1,s.y −1),t)
		
		// check boundaries 
		if(i<0 || j<0 || k>=r.size() || l>=r[0].size())
				return 0;

		// compute multiplicity
		int res =  r[i][j][k][l];
		if (k+1 < r.size())
				res -= r[i][j][k+1][l];
		if (l+1 < r[0].size())
				res -= r[i][j][k][l+1];
		if (k+1 < r.size() && l+1 < r[0].size())
				res += r[i][j][k+1][l+1];

		return res;
}


void compute_R_S_incl_excl(const rank& r, barcode& b) {
// The formula is m(s,t) = m(s,t+) − m(s,(t.x +1,t.y)+) − m(s,(t.x,t.y +1)+) + m(s,(t.x +1,t.y +1)+)
		
		for (int i = 0; i<r.size(); i++){
				for (int j = 0; j<r[0].size(); j++){
						for (int k = i; k<r.size(); k++){
								for (int l = j; l<r[0].size(); l++) {
										int m = multp(r, i, j, k, l) - multp(r, i-1, j, k, l)
														- multp(r, i, j-1, k, l) + multp(r, i-1, j-1, k, l);
										if (m != 0) {
												bar bb(entry(i,j), entry(k,l));
												b[bb] = m;
										}
								}
						}
				}
		}
}


int main(int argc, char* const argv[]) {
		int x_range;
		int y_range;
		int rank_dim;
		string filename;
		string out_filename;

		if(argc!=6){
			cout <<"Usage: " << argv[0] << " input_filename range_x range_y rank_dim output_filename\n";
				return 0;
		}
		else{
				filename=argv[1];
				x_range=std::stoi(argv[2])-1;
				y_range=std::stoi(argv[3])-1;
				rank_dim=std::stoi(argv[4]);
				out_filename=argv[5];
		}

		vector<float> x_values;
		vector<float> y_values;
		vector<simplex_node> simplices;
		
		if (!read_input_filtration(filename, rank_dim, simplices, x_values, y_values))
			return 0;

		vector<simplex_node> simplices_integer=simplices; // book-keeping: keep original simplex grades
		discretize_filtration(simplices_integer, x_values, y_values, x_range, y_range);
		
		rank rank_inv;
		zero_rank(rank_inv,x_range+1,y_range+1);
		compute_rank_invariant (simplices_integer, x_range, y_range, rank_dim, rank_inv);    

		barcode b;
		compute_R_S_incl_excl(rank_inv,b);

		cout << "\nbarcode size: " << b.size() << std::endl;

		std::cout << "writing output barcode to " << out_filename << std::endl;
		std::map<bar, int>::iterator it;
		std::ofstream f(out_filename);

		f << x_range << " "<< y_range<<endl;
		for(it=b.begin(); it != b.end(); it++)
			f << (it->first).first.first  << " " << (it->first).first.second << " " <<  (it->first).second.first << " " << (it->first).second.second << " " <<   it->second <<  endl;

		return 0;
}


