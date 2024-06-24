#include <iostream>
#include <cmath>

float mean_squarred_error(float m,float b, float x[],float y[]){
    float total_error {0};
    for (int i = 0; i < 6;i++){
        total_error += pow((y[i] - (m * x[i] + b)),2); 
    }
    return total_error / 6;
}

float gradient_descent(float m_now,float b_now,float x[],float y[],float learning_rate,   float *m , float *b){
    float m_gradient {0};
    float b_gradient {0};
    //float m,b {0};

    float len {95};

    for (int i = 0; i < len; i++){
        m_gradient += -(2/len) * x[i] * (y[i] - (m_now * x[i] + b_now));
        b_gradient += -(2/len) * (y[i] - (m_now * x[i] + b_now));
    }

    *m = m_now - (m_gradient * learning_rate);
    *b = b_now - (b_gradient  * learning_rate);

    //std::cout << "M = " << *m_now << " , B = " << *b_now << std::endl;

    return 0; 
}





int main(){
    // float area[] = {260.0,300.0,320.0,360.0,400.0,410};
    // float targets[] = {550.0,565.0,610.0,595.0,760.0,810.0};

    float area[] = {5.5277,8.5186,7.0032,5.8598,8.3829,7.4764,8.5781,6.4862,5.0546,5.7107,4.1640,5.7340,8.4084,5.6407,5.3794,6.3654,5.1301,6.4296,7.0708,6.1891,0.2700,5.4901,6.3261,5.5649,8.9450,2.8280,0.9570,3.1760,2.2030,5.2524,6.5894,9.2482,5.8918,8.2111,7.9334,8.0959,5.6063,2.8360,6.3534,5.4069,6.8825,1.7080,5.7737,7.8247,7.0931,5.0702,5.8014,1.7000,5.5416,7.5402,5.3077,7.4239,7.6031,6.3328,6.3589,6.2742,5.6397,9.3102,9.4536,8.8254,5.1793,1.2790,4.9080,8.9590,7.2182,8.2951,0.2360,5.4994,0.3410,0.1360,7.3345,6.0062,7.2259,5.0269,6.5479,7.5386,5.0365,0.2740,5.1077,5.7292,5.1884,6.3557,9.7687,6.5159,8.5172,9.1802,6.0020,5.5204,5.0594,5.7077,7.6366,5.8707,5.3054,8.2934,3.3940,5.4369};

    float targets[] = {9.13020,3.66200,1.85400,6.82330,1.88600,4.34830,2.00000,6.59870,3.81660,3.25220,5.50500,3.15510,7.22580,0.71618,3.51290,5.30480,0.56077,3.65180,5.38930,3.13860,1.76700,4.26300,5.18750,3.08250,2.63800,3.50100,7.04670,4.69200,4.14700,1.22000,5.99660,2.13400,1.84950,6.54260,4.56230,4.11640,3.39280,0.11700,5.49740,0.55657,3.91150,5.38540,2.44060,6.73180,1.04630,5.13370,1.84400,8.00430,1.01790,6.75040,1.83960,4.28850,4.99810,1.42330,1.42110,2.47560,4.60420,3.96240,5.41410,5.16940,0.74279,7.92900,2.05400,7.05400,4.88520,5.74420,7.77540,1.01730,0.99200,6.67990,4.02590,1.27840,3.34110,2.68070,0.29678,3.88450,5.70140,6.75260,2.05760,0.47953,0.20421,0.67861,7.54350,5.34360,4.24150,6.79810,0.92695,0.15200,2.82140,1.84510,4.29590,7.20290,1.98690,0.14454,9.05510,0.61705};

    std::cout << "MSE = "<< mean_squarred_error(100,100,area,targets) << std::endl;

    float m {0};
    float b {0};
    float L {0.0001};
    int epochs {50000};

    float result1 {0.0};
    float result2 {0.0};


    std::cout << "M = " << m << " , B = " << b << std::endl;


    //gradient_descent(&m,&b,area,targets,L);

    for (int i = 0; i <= epochs; i++){
        gradient_descent(m,b,area,targets,L,&result1,&result2);

        //test 
        // m = result1;
        // b = result2;
    }

    std::cout << "M = " << m << " , B = " << b << std::endl;
    std::cout << "M = " << result1 << " , B = " << result2 << std::endl;
}

/*
m, b doesn't update properly in the function

differentiate m,b variables (parsed to the function && used in the function) | dependent

using tuples
using pointers
*/