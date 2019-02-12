#include <api/components.h>
#include <api/kvindex_itf.h>
#include <iostream>
#include "../src/ramrbtree.h"

using namespace std;

int main(){
    RamRBTree tree;

    tree.insert("abc");
    tree.insert("efg");
    cout<<tree.get(0)<<endl;
    cout<<tree.get(1)<<endl;
    cout<<tree.count()<<endl;
    tree.get(3);
    return 0;
}

