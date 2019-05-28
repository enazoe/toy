#include "selectivate_sort.hpp"
#include "incert_sort.hpp"
#include <iostream>
#include <vector>

int main(int argc, char const *argv[])
{
    IncertSort sort;
    std::vector<int> arr= {3,4,2,1,0,4};
  
    sort.incert_sort(arr);
    for(int i=0;i<arr.size();++i)
    {
        std::cout<<arr[i]<<std::endl;
    }
    return 0;
}
