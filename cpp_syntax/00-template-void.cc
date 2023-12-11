# include<iostream>

template <typename T, typename Enable = void>
class CTmpl {
public:
    T FunA(T a, T b);
    CTmpl() {
         std::cout << "CTmpl<T, 1111" << std::endl;
    }
};

template <typename T>
class CTmpl <T, typename T::StreamkFeature> {
public:
    T FunA(T a, T b);
    CTmpl() {
        std::cout << "CTmpl<T, typename T::StreamkFeature>" << std::endl;
    };
};

class Haha {
public:
    typedef  int StreamkFeature;
    Haha() {  };
};

int main() {
  CTmpl<Haha, Haha::StreamkFeature> a;
  CTmpl<Haha, double> b;
  CTmpl<Haha, int> c;
  CTmpl<int> d;
  CTmpl<Haha, int> e;
  CTmpl<Haha> f;

  return 0;
}

