#include<iostream>
#include<time.h>
#include<pmmintrin.h>
#include<xmmintrin.h>
#include <immintrin.h>

#define N 1024
#define count 1
using namespace std;


float A[N][N];

void ori_LU() {
    for (int k = 0; k < N; k++){
        float tmp =A[k][k];
        for (int j = k; j < N; j++){
            A[k][j] = A[k][j] / tmp;
        }
        for (int i = k + 1; i < N; i++){
            float tmp2 = A[i][k];
            for (int j = k + 1; j < N; j++){
                A[i][j] = A[i][j] - tmp2 * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}



void SSE_LU(){
    __m128 t1, t2, t3, t4;
    for (int k = 0; k < N; k++){
        float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
        t1 = _mm_loadu_ps(tmp);
        for (int j = N - 4; j >=k; j -= 4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t3);
        }
        if (k % 4 != (N % 4)){
            for (int j = k; j % 4 != ( N% 4); j++){
                A[k][j] = A[k][j] / tmp[0];
            }
        }
        for (int j = (N % 4) - 1; j>= 0; j--){
            A[k][j] = A[k][j] / tmp[0];
        }
        for (int i = k + 1; i < N; i++){
            float tmp[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
            t1 = _mm_loadu_ps(tmp);
            for (int j = N - 4; j >k;j -= 4){
                t2 = _mm_loadu_ps(A[i] + j);
                t3 = _mm_loadu_ps(A[k] + j);
                t4 = _mm_sub_ps(t2,_mm_mul_ps(t1, t3));
                _mm_storeu_ps(A[i] + j, t4);
            }
            for (int j = k + 1; j % 4 !=(N % 4); j++){
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void AVX_LU(){
    __m256 t1, t2, t3, t4;
    for (int k = 0; k < N; k++){
        float tmp[8];
        for(int i=0;i<8;i++)tmp[i]=A[k][k];
        t1 = _mm256_loadu_ps(tmp);
        for (int j = N - 8; j >=k; j -= 8) {
            t2 = _mm256_loadu_ps(A[k] + j);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(A[k] + j, t3);
        }
        if (k % 8 != (N % 8)){
            for (int j = k; j % 8 != ( N% 8); j++){
                A[k][j] = A[k][j] / tmp[0];
            }
        }
        for (int j = (N % 8) - 1; j>= 0; j--){
            A[k][j] = A[k][j] / tmp[0];
        }
        for (int i = k + 1; i < N; i++){
            float tmp[8] ;
            for(int i=0;i<8;i++)tmp[i]=A[i][k];
            t1 = _mm256_loadu_ps(tmp);
            for (int j = N - 8; j >k;j -= 8){
                t2 = _mm256_loadu_ps(A[i] + j);
                t3 = _mm256_loadu_ps(A[k] + j);
                t4 = _mm256_sub_ps(t2,_mm256_mul_ps(t1, t3));
                _mm256_storeu_ps(A[i] + j, t4);
            }
            for (int j = k + 1; j % 8 !=(N % 8); j++){
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}


void m_reset(){
    for(int i = 0; i < N ;i++){
        for(int j = 0;j<i;j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for(int j = i + 1;j<N;j++)
            A[i][j] = rand();
    }
    for(int k = 0;k<N;k++)
        for(int i = k+1;i<N;i++)
            for(int j =0;j<N;j++)
                A[i][j]+=A[k][j];
}

int main()
{

    clock_t  clockBegin,clockEnd;
    m_reset();
    clockBegin = clock();
    for(int i = 0;i<count;i++)
    ori_LU();
    clockEnd = clock();
    float standard = clockEnd - clockBegin;
    cout << "ori_LU =  " << standard << "ms" << endl;

    m_reset();
    clockBegin = clock();
    for(int i = 0;i<count;i++)
    SSE_LU();
    clockEnd = clock();
    float SSE = clockEnd - clockBegin;
    cout << "SSE_LU =  " << clockEnd - clockBegin << "ms" << endl;
    cout<<"Ratio of Acc = "<<standard/SSE<<endl;

    m_reset();
    clockBegin = clock();
    for(int i = 0;i<count;i++)
    AVX_LU();
    clockEnd = clock();
    float AVX = clockEnd - clockBegin;
    cout << "AVX_LU =  " << clockEnd - clockBegin << "ms" << endl;
    cout<<"Ratio of Acc = "<<standard/AVX<<endl;
    system("pause");
    return 0;

}
