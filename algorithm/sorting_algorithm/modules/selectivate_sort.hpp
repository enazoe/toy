#ifndef SELECTIVATE_SORT_HPP_
#define SELECTIVATE_SORT_HPP_
#include <vector>

class SselectSort
{
public:
	SselectSort()
	{
		
	}

	~SselectSort()
	{

	}

	void select_sort(std::vector<int> &vec_arr)
	{
		for(int i=0;i<vec_arr.size();++i)
		{
			int min=i;
			for(int j=i+1;j<vec_arr.size();++j)
			{
				if(vec_arr[j]<vec_arr[i])
				{
					min = j;
				}
			}

			int key = vec_arr[min];
			while(min>i)
			{
				vec_arr[min]=vec_arr[min-1];
				min--;
			}
			vec_arr[i]=key;
		}
	}

};

#endif