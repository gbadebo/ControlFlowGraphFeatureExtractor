
 
int add(int a[], int size)
{
    int i, sum = 0;
    for ( i = 0 ; i < size ; i++ ) sum += a[i];
    return sum;
}
 
int main ( ) {
    int arr[5];
    input(arr,5);
    printf("The sum is: %d\n", add(arr,5) );
    return 0;
}
