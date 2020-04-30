/**
 * Compile by:
 * $ gcc benchmark_rtm.c -fopenmp -mrtm
 * 
 * collision probability (chance of a write access) needs to be given to the benchmark at the start of the program as argument (e.g.: ./a.out 5)
 * 
 * A CSV-file can be created with the duration of each iteration.
 * Look for line "create_file(time_values);" (with strg + f) and remove comment "//" to run the function that creates CSV file with name given at #define DATANAME
 * 
 * useful variables that can be changed to create certain conditions:
 * 
 * RETRIES 					--> number of retries, before fallback to locks
 * SIZE_X and SIZE_Y		--> size of the shared data (2-d-array)
 * WORK_CYCLES_CRITICAL 	--> average number of cycles done in critical section
 * WORK_CYCLES_NONCRITICAL	--> average number of cycles done after critical section before starting new iteration
 * THREADS					--> number of threads (only for measurement! Look at comment behind "#define THREADS"-statement)
 * 
 */
 
#include <stdio.h>
#include <omp.h>
#include "immintrin.h"
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdatomic.h>
#include <pthread.h>

#ifdef SCOREP_USER_ENABLE
#include <scorep/SCOREP_User.h>
#endif

#define _ABORT_LOCK_BUSY 0xff

#define USE_RTM

#define ITERATIONS 100000	//number of iterations

#define RETRIES 4

#define SIZE_X 8		//8 int64 values per cacheline
#define SIZE_Y 2	//SIZE_X * SIZE_Y = number of int64-values (up to 512 cachlines in L1)

#define START 0			//start of time measurement
#define STOP 1			//end of time measurement

#define WORK_CYCLES_CRITICAL 50000
#define WORK_CYCLES_NONCRITICAL 50000

#define THREADS 8		//number of threads is set with "export OMP_NUM_THREADS=x" --> this number is for measurment
#define DATANAME "data.csv"

#define pause() asm volatile("pause" ::: "memory")

int64_t data[SIZE_X][SIZE_Y] = {0};		//Data, the benchmark does its calculations with

static void work(int critical);
static inline unsigned long long rdtscl(void);
static inline void create_file(long long data[THREADS][2][ITERATIONS]);
//static void transaction_work(int random_number, int work_sum_critical, int prob_collision, int i, int64_t *sum);

//static volatile atomic_flag my_lock_flag;

#ifdef SCOREP_USER_ENABLE
SCOREP_USER_REGION_DEFINE( lock_region_handle );
SCOREP_USER_REGION_DEFINE( rtm_region_handle );
#endif


static inline int my_lock(pthread_spinlock_t *spin_l)
{
#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_ENTER(lock_region_handle);
#endif
	//while (atomic_flag_test_and_set(&my_lock_flag));
	//omp_set_lock(lock);
	pthread_spin_lock(spin_l);
}

static inline int my_unlock(pthread_spinlock_t *spin_l)
{
	pthread_spin_unlock(spin_l);
	//omp_unset_lock(lock);
	//atomic_flag_clear(&my_lock_flag);
#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_END(lock_region_handle);
#endif
}

int main(int argc, char* argv[]) {

        //omp_lock_t lock;
        //omp_init_lock(&lock);

	pthread_spinlock_t spin_l;
	pthread_spin_init(&spin_l, 0);
/*
	int a;
	a = (int)spin_l;
	printf("1: %d\n", a);

	my_lock(&spin_l);
	a = (int)spin_l;
        printf("2: %d\n", a);
	my_unlock(&spin_l);
	a = (int)spin_l;
	printf("3: %d\n", a);

*/

#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_INIT( lock_region_handle, "lock test-and-set",SCOREP_USER_REGION_TYPE_COMMON )
	SCOREP_USER_REGION_INIT( rtm_region_handle, "lock rtm",SCOREP_USER_REGION_TYPE_COMMON )
#endif
	//atomic_flag_clear(&my_lock_flag);

        if (argc <= 1)
        {
			printf("Please specify the probability of a collision caused by one thread.");
			return 1;
        }

        int prob_collision = atoi(argv[1]);			//Probability of 1 thread to cause a collision
	long long time_values[THREADS][2][ITERATIONS];

        #pragma omp parallel
        //#pragma omp parallel num_threads(number_of_threads)
        {

			int64_t sum = 0;
			int id = omp_get_thread_num();			//get thread-id for TSC measurment
			unsigned int seed = time(NULL) + (id * 7);	//create thread-local seed for rand_r()
			int random_number;
			int work1;
			int work2;
			int work_sum_crit;
			int work_sum_noncrit;

			int retries = 0;

			//do some work so the important stuff does not start at the same time for every thread

			for (int k = 0; k <= id; k++)
			{
				work(WORK_CYCLES_CRITICAL);
			}

			for (int i = 0; i < ITERATIONS; i++)
			{
				//random number in range [1, 100] to realize probability for collision
				random_number = (rand_r(&seed) % 100) + 1;

				//additional work that is done in- and outside of critical section should be a varying number of cycles
				work1 = (rand_r(&seed) % (WORK_CYCLES_CRITICAL / 2));
				work2 = (rand_r(&seed) % (WORK_CYCLES_CRITICAL / 2));
				work_sum_crit = work1 + work2;

				work1 = (rand_r(&seed) % (WORK_CYCLES_NONCRITICAL / 2));
				work2 = (rand_r(&seed) % (WORK_CYCLES_NONCRITICAL / 2));
				work_sum_noncrit = work1 + work2;


				//start time measurement
				time_values[id][START][i] = rdtscl();

#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_ENTER(rtm_region_handle);
#endif

#ifdef USE_RTM
				while(1)
				{
					//start transaction
					unsigned status = _xbegin();
					if (status == _XBEGIN_STARTED)
					{
						if ((int)spin_l != 1)
						{
							_xabort(_ABORT_LOCK_BUSY);
						}

				        if (random_number <= prob_collision)            //leads to a conflict with given probability
       					{
                				data[i%SIZE_X][i%SIZE_Y] += 1;          //possible conflict
                				work(work_sum_crit);
        				}
        				else
        				{
                				sum += data[i%SIZE_X][i%SIZE_Y];       //no conflict, just read data and write its own sum
                				work(work_sum_crit);
        				}


					_xend();	//end of transaction. if successful commited, break out of loop. if not successful commited, restart transaction or use locks

#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_END(rtm_region_handle);
#endif

					break;
					}

					// if (there is a reason for a restart) --> restart
					// else --> use locks instead
					if (!(status & _XABORT_RETRY)
                				&& !(status & _XABORT_CONFLICT)
                				&& !((status & _XABORT_EXPLICIT)
                     				&& _XABORT_CODE(status) != _ABORT_LOCK_BUSY)
						&& (retries < RETRIES))
					{
						while ((int)spin_l != 1)	//wait for spin to be free before restart
						{
							pause();
						}
						retries++;
						continue;
					}
					else
					{
#endif
						//fallback to spinlock
						//pthread_spin_lock(&spin_l);
						retries = 0;
						my_lock(&spin_l);
					        if (random_number <= prob_collision)            //leads to a conflict with given probability
        					{
                					data[i%SIZE_X][i%SIZE_Y] += 1;          //possible conflict
                					work(work_sum_crit);
        					}
        					else
        					{
                					sum += data[i%SIZE_X][i%SIZE_Y];       //no conflict, just read data and write its own sum
                					work(work_sum_crit);
        					}


						my_unlock(&spin_l);
						//pthread_spin_unlock(&spin_l);

#ifdef USE_RTM

#ifdef SCOREP_USER_ENABLE
	SCOREP_USER_REGION_END(rtm_region_handle);
#endif

						break;
					}
				}
#endif
				work(work_sum_noncrit); //work outside the lock

				//stop time measurement
				time_values[id][STOP][i] = rdtscl();


			} //iterations

			printf("process %d finished with sum %ld, %d collision probability\n", id, sum, prob_collision);

		} //omp parallel

		//create_file(time_values);
		pthread_spin_destroy(&spin_l);
} //main

/*
static void transaction_work(int random_number, int work_sum_crit, int prob_collision, int i, int64_t *sum) {

	if (random_number <= prob_collision)		//leads to a conflict with given probability
	{
		data[i%SIZE_X][i%SIZE_Y] += 1;		//possible conflict
		work(work_sum_crit);
	}
	else
	{
		*sum += data[i%SIZE_X][i%SIZE_Y];	//no conflict, just read data and write its own sum
		work(work_sum_crit);
	}
	return;
}
*/
static void work(int work_cycles) {

	//long long int work_cycles_critical = 10000;
	//long long int work_cycles_not_critical = 10000;

	long long int start = rdtscl(); //number of cycles done since reset

	int variable_for_work = 0;

	while ((start + work_cycles) > rdtscl())
	{
		variable_for_work++;
	}
	return;
}

static inline unsigned long long rdtscl(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

static inline void create_file (long long data[THREADS][2][ITERATIONS]) {
	FILE *fp;
	fp = fopen(DATANAME, "w+");
	long long int duration;

	//fprintf(fp, "DURATION\n");

	for (int i = 2; i < ITERATIONS; i++) {
		for (int j = 0; j < THREADS; j++) {
			duration = data[j][STOP][i] - data[j][START][i];
			if (j == 0) {
				fprintf(fp, "%lli", duration);
			}
			else {
				fprintf(fp, ", %lli", duration);
			}
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	printf("file created");
}
