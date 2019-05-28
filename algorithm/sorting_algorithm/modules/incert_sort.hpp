#ifndef INCERT_SORT_HPP_
#define INCERT_SORT_HPP_
#include <vector>

class IncertSort
{
public:
	IncertSort()
	{
		
	}

	~IncertSort()
	{

	}

	void incert_sort(std::vector<int> &vec_arr)
	{
		for(int i=1;i<vec_arr.size();++i)
        {
            int key = vec_arr[i];
            int j = i-1;
            while(j>=0&&key<vec_arr[j])
            {
                vec_arr[j+1]=vec_arr[j];
                j--;
            }
            vec_arr[j+1]=key;
        }
	}

};

#endif