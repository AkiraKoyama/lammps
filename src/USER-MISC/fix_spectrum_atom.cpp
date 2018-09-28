/* ------------------
 * {
 *
 *
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "fix_spectrum_atom.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{X,V,F,COMPUTE,FIX,VARIABLE};
enum{LIN,LOG};

#define INVOKED_PERATOM 8

/* ---------------------------------------------------------------------- */

FixSpectrumAtom::FixSpectrumAtom(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  nvalues(0), which(NULL), argindex(NULL), value2index(NULL),
  ids(NULL), array(NULL), v_omega_r(NULL), v_omega_i(NULL), fp(NULL)
{
  MPI_Comm_rank(world,&me);
  if (narg < 10) error->all(FLERR,"Illegal fix spectrum/atom command");

  nevery = force->inumeric(FLERR,arg[3]);
  neval = force->inumeric(FLERR,arg[4]);
  omega_lo = force->numeric(FLERR,arg[5]);
  omega_hi = force->numeric(FLERR,arg[6]);
  omega_n = force->inumeric(FLERR,arg[7]);
  if (strcmp(arg[8],"linear") == 0)
    omega_scale=LIN;
  else if (strcmp(arg[8],"log") == 0)
    omega_scale=LOG;
  else
    error->all(FLERR,"Illegal fix spectrum/atom command");

  // expand args if any have wildcard character "*"
  // this can reset nvalues

  int expand = 0;
  char **earg;
  int nargnew = input->expand_args(narg-9,&arg[9],1,earg);

  if (earg != &arg[9]) expand = 1;
  arg = earg;

  // parse values

  which = new int[nargnew];
  argindex = new int[nargnew];
  ids = new char*[nargnew];
  value2index = new int[nargnew];
  nvalues = 0;

  int iarg = 0;;

  while (iarg < nargnew) {
    ids[iarg] = NULL;
    if (strcmp(arg[iarg],"x") == 0 ||
      strcmp(arg[iarg],"y") == 0 ||
      strcmp(arg[iarg],"z") == 0 ||
      strcmp(arg[iarg],"vx") == 0 ||
      strcmp(arg[iarg],"vy") == 0 ||
      strcmp(arg[iarg],"vz") == 0 ||
      strcmp(arg[iarg],"fx") == 0 ||
      strcmp(arg[iarg],"fy") == 0 ||
      strcmp(arg[iarg],"fz") == 0) {
      if (strcmp(arg[iarg],"x") == 0) {
        which[iarg] = X;
        argindex[iarg] = 0;
      } else if (strcmp(arg[iarg],"y") == 0) {
        which[iarg] = X;
        argindex[iarg] = 1;
      } else if (strcmp(arg[iarg],"z") == 0) {
        which[iarg] = X;
        argindex[iarg] = 2;

      } else if (strcmp(arg[iarg],"vx") == 0) {
        which[iarg] = V;
        argindex[iarg] = 0;
      } else if (strcmp(arg[iarg],"vy") == 0) {
        which[iarg] = V;
        argindex[iarg] = 1;
      } else if (strcmp(arg[iarg],"vz") == 0) {
        which[iarg] = V;
        argindex[iarg] = 2;

      } else if (strcmp(arg[iarg],"fx") == 0) {
        which[iarg] = F;
        argindex[iarg] = 0;
      } else if (strcmp(arg[iarg],"fy") == 0) {
        which[iarg] = F;
        argindex[iarg] = 1;
      } else if (strcmp(arg[iarg],"fz") == 0) {
        which[iarg] = F;
        argindex[iarg] = 2;
      }
      nvalues++;
      iarg++;
    } else if (strncmp(arg[iarg],"c_",2) == 0 ||
               strncmp(arg[iarg],"f_",2) == 0 ||
               strncmp(arg[iarg],"v_",2) == 0) {
      if (arg[iarg][0] == 'c') which[iarg] = COMPUTE;
      else if (arg[iarg][0] == 'f') which[iarg] = FIX;
      else if (arg[iarg][0] == 'v') which[iarg] = VARIABLE;

      int n = strlen(arg[iarg]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[iarg][2]);

      char *ptr = strchr(suffix,'[');
      if (ptr) {
        if (suffix[strlen(suffix)-1] != ']')
          error->all(FLERR,"Illegal fix spectrum/atom command");
        argindex[iarg] = atoi(ptr+1);
        *ptr = '\0';
      } else argindex[iarg] = 0;

      n = strlen(suffix) + 1;
      ids[iarg] = new char[n];
      strcpy(ids[iarg],suffix);
      delete [] suffix;

      nvalues++;
      iarg++;
    } else break;
  }

  overwrite = 0;

  while (iarg < nargnew)
  {
    if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 > nargnew) error->all(FLERR,"Illegal fix spectrum/atom command");
      if (me == 0) {
        fp = fopen(arg[iarg+1],"w");
        if (fp == NULL) {
          char str[128];
          snprintf(str,128,"Cannot open fix spectrum/atom file %s",arg[iarg+1]);
          error->one(FLERR,str);
        }
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"overwrite") == 0) {
      overwrite = 1;
      iarg += 1;
    } else error->all(FLERR,"Illegal fix spectrum/atom command");
  }

  // if wildcard expansion occurred, free earg memory from exapnd_args()

  if (expand) {
    for (int i = 0; i < nvalues; i++) delete [] earg[i];
    memory->sfree(earg);
  }

  // setup and error check

  if (omega_lo >= omega_hi)
    error->all(FLERR,"Illegal fix spectrum/atom command");
  if (omega_n <2)
    error->all(FLERR,"Illegal fix spectrum/atom command");

  for (int i = 0; i < nvalues; i++) {
    if (which[i] == COMPUTE) {
      int icompute = modify->find_compute(ids[i]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix spectrum/atom does not exist");
      if (modify->compute[icompute]->peratom_flag == 0)
        error->all(FLERR,
                   "Fix spectrum/atom compute does not calculate per-atom values");
      if (argindex[i] == 0 &&
          modify->compute[icompute]->size_peratom_cols != 0)
        error->all(FLERR,"Fix spectrum/atom compute does not "
                   "calculate a per-atom vector");
      if (argindex[i] && modify->compute[icompute]->size_peratom_cols == 0)
        error->all(FLERR,"Fix spectrum/atom compute does not "
                   "calculate a per-atom array");
      if (argindex[i] &&
          argindex[i] > modify->compute[icompute]->size_peratom_cols)
        error->all(FLERR,"Fix spectrum/atom compute array is accessed out-of-range");

    } else if (which[i] == FIX) {
      int ifix = modify->find_fix(ids[i]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix spectrum/atom does not exist");
      if (modify->fix[ifix]->peratom_flag == 0)
        error->all(FLERR,"Fix spectrum/atom fix does not calculate per-atom values");
      if (argindex[i] == 0 && modify->fix[ifix]->size_peratom_cols != 0)
        error->all(FLERR,
                   "Fix spectrum/atom fix does not calculate a per-atom vector");
      if (argindex[i] && modify->fix[ifix]->size_peratom_cols == 0)
        error->all(FLERR,
                   "Fix spectrum/atom fix does not calculate a per-atom array");
      if (argindex[i] && argindex[i] > modify->fix[ifix]->size_peratom_cols)
        error->all(FLERR,"Fix spectrum/atom fix array is accessed out-of-range");
      if (nevery % modify->fix[ifix]->peratom_freq)
        error->all(FLERR,
                   "Fix for fix spectrum/atom not computed at compatible time");

    } else if (which[i] == VARIABLE) {
      int ivariable = input->variable->find(ids[i]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix spectrum/atom does not exist");
      if (input->variable->atomstyle(ivariable) == 0)
        error->all(FLERR,"Fix spectrum/atom variable is not atom-style variable");
    }
  }

  if (fp && me == 0) {
    clearerr(fp);
    fprintf (fp,"omega [1/time]");
    for (int k = 0; k < nvalues; k++)
      fprintf (fp," P_%s", arg[k]);
    fprintf(fp,"\n");
    if (ferror(fp))
      error->one(FLERR,"Error writing file header");

    filepos = ftell(fp);
  }


  // this fix produces a global array

  array_flag = 1;
  size_array_rows = omega_n;
  size_array_cols = nvalues+1;
  extarray = 0;
  memory->create(array,omega_n,nvalues,"fix_spectrum/atom:array");

  // perform initial allocation of atom-based array
  // register with Atom class

  grow_arrays(atom->nmax);
  atom->add_callback(0);
  for (int i = 0; i < omega_n; i++)
    for (int j= 0; j < nvalues; j++)
      for (int n=0; n < atom->nlocal; n++) {
        v_omega_r[n][j][i] = 0.;
        v_omega_i[n][j][i] = 0.;
      }

  // zero the array since dump may access it on timestep 0
  // zero the array since a variable may access it before first run

  for (int i = 0; i < omega_n; i++)
    for (int m = 0; m < nvalues; m++)
      array[i][m] = 0.0;

  count = 0;

  // nvalid = next step on which end_of_step does something
  // add nvalid to all computes that store invocation times
  // since don't know a priori which are invoked by this fix
  // once in end_of_step() can set timestep for ones actually invoked

  nvalid_last = -1;
  nvalid = update->ntimestep + nevery;
  modify->addstep_compute_all(nvalid);
}

/* ---------------------------------------------------------------------- */

FixSpectrumAtom::~FixSpectrumAtom()
{
  // unregister callback to this fix from Atom class

  atom->delete_callback(id,0);

  delete [] which;
  delete [] argindex;
  for (int m = 0; m < nvalues; m++) delete [] ids[m];
  delete [] ids;
  delete [] value2index;
  delete [] omega_vec;

  memory->destroy(array);
  memory->destroy(v_omega_r);
  memory->destroy(v_omega_i);

  if (fp && me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixSpectrumAtom::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSpectrumAtom::init()
{
  // set indices and check validity of all computes,fixes,variables

  for (int m = 0; m < nvalues; m++) {
    if (which[m] == COMPUTE) {
      int icompute = modify->find_compute(ids[m]);
      if (icompute < 0)
        error->all(FLERR,"Compute ID for fix spectrum/atom does not exist");
      value2index[m] = icompute;

    } else if (which[m] == FIX) {
      int ifix = modify->find_fix(ids[m]);
      if (ifix < 0)
        error->all(FLERR,"Fix ID for fix spectrum/atom does not exist");
      value2index[m] = ifix;

    } else if (which[m] == VARIABLE) {
      int ivariable = input->variable->find(ids[m]);
      if (ivariable < 0)
        error->all(FLERR,"Variable name for fix spectrum/atom does not exist");
      value2index[m] = ivariable;

    } else value2index[m] = -1;
  }

  // frequency range

  omega_vec = new double[omega_n];
  if (omega_scale == LIN) {
    double d_omega=(omega_hi-omega_lo)/(omega_n-1);
    for (int k=0; k<omega_n; k++) omega_vec[k] = omega_lo+k*d_omega;
  } else {
    double d_omega=log(omega_hi/omega_lo)/(omega_n-1);
    for (int k=0; k<omega_n; k++) omega_vec[k] = omega_lo*exp(k*d_omega);
  }




  // need to reset nvalid if nvalid < ntimestep b/c minimize was performed

  if (nvalid < update->ntimestep) {
    nvalid = update->ntimestep + nevery;
    modify->addstep_compute_all(nvalid);
  }
}

/* ----------------------------------------------------------------------
   only does something if nvalid = current timestep
------------------------------------------------------------------------- */

void FixSpectrumAtom::setup(int /*vflag*/)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixSpectrumAtom::end_of_step()
{
  int i,j,m,n;

  // skip if not step which requires doing something
  // error check if timestep was reset in an invalid manner

  bigint ntimestep = update->ntimestep;
  if (ntimestep < nvalid_last || ntimestep > nvalid)
    error->all(FLERR,"Invalid timestep reset for fix spectrum/atom");
  if (ntimestep != nvalid) return;
  nvalid_last = nvalid;

  // zero if first step

  if (count == 0)
    for (i = 0; i < omega_n; i++)
      for (j= 0; j < nvalues; j++)
        for (n=0; n < atom->nlocal; n++)
        {
          v_omega_r[n][j][i] = 0.;
          v_omega_i[n][j][i] = 0.;
        }

  // accumulate results of attributes,computes,fixes,variables to local copy
  // compute/fix/variable may invoke computes so wrap with clear/add

  modify->clearstep_compute();

  int *mask = atom->mask;

  double *c_t = new double[omega_n];
  double *s_t = new double[omega_n];
  for (i=0; i<omega_n; i++) {
    c_t[i] = cos(omega_vec[i]*count*nevery*update->dt);
    s_t[i] = sin(omega_vec[i]*count*nevery*update->dt);
  }

  int nlocal = atom->nlocal;
  for (m = 0; m < nvalues; m++) {
    n = value2index[m];
    j = argindex[m];

    if (which[m] == X) {
      double **x = atom->x;
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          for (n=0; n<omega_n; n++) {
            v_omega_r[i][m][n] += c_t[n]*x[i][j];
            v_omega_i[i][m][n] += s_t[n]*x[i][j];
          }

    } else if (which[m] == V) {
      double **v = atom->v;
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          for (n=0; n<omega_n; n++) {
            v_omega_r[i][m][n] += c_t[n]*v[i][j];
            v_omega_i[i][m][n] += s_t[n]*v[i][j];
          }

    } else if (which[m] == F) {
      double **f = atom->f;
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          for (n=0; n<omega_n; n++) {
            v_omega_r[i][m][n] += c_t[n]*f[i][j];
            v_omega_i[i][m][n] += s_t[n]*f[i][j];
          }

    // invoke compute if not previously invoked

    } else if (which[m] == COMPUTE) {
      Compute *compute = modify->compute[n];
      if (!(compute->invoked_flag & INVOKED_PERATOM)) {
        compute->compute_peratom();
        compute->invoked_flag |= INVOKED_PERATOM;
      }

      if (j == 0) {
        double *compute_vector = compute->vector_atom;
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            for (n=0; n<omega_n; n++) {
              v_omega_r[i][m][n] += c_t[n]*compute_vector[i];
              v_omega_i[i][m][n] += s_t[n]*compute_vector[i];
            }
      } else {
        int jm1 = j - 1;
        double **compute_array = compute->array_atom;
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            for (n=0; n<omega_n; n++) {
              v_omega_r[i][m][n] += c_t[n]*compute_array[i][jm1];
              v_omega_i[i][m][n] += s_t[n]*compute_array[i][jm1];
            }
      }

    // access fix fields, guaranteed to be ready

    } else if (which[m] == FIX) {
      if (j == 0) {
        double *fix_vector = modify->fix[n]->vector_atom;
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            for (n=0; n<omega_n; n++) {
              v_omega_r[i][m][n] += c_t[n]*fix_vector[i];
              v_omega_i[i][m][n] += s_t[n]*fix_vector[i];
            }
      } else {
        int jm1 = j - 1;
        double **fix_array = modify->fix[n]->array_atom;
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            for (n=0; n<omega_n; n++) {
              v_omega_r[i][m][n] += c_t[n]*fix_array[i][jm1];
              v_omega_i[i][m][n] += s_t[n]*fix_array[i][jm1];
            }
      }

    // evaluate atom-style variable
    // final argument = 1 sums result to array

    } else if (which[m] == VARIABLE) {
      double *var_vec = new double[atom->nmax];
      input->variable->compute_atom(n,igroup,var_vec,nvalues,0);
      for (i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
          for (n=0; n<omega_n; n++) {
            v_omega_r[i][m][n] += c_t[n]*var_vec[i];
            v_omega_i[i][m][n] += s_t[n]*var_vec[i];
          }
      delete [] var_vec;
    }
  }
  count += 1;
  delete [] c_t;
  delete [] s_t;

  for (i = 0; i < omega_n; i++)
    for (j= 0; j < nvalues; j++)
      array[i][j] = 0;

  if (count*nevery % neval == 0)
  {
    double **array_loc;
    memory->create(array_loc,omega_n,nvalues,"fix_spectrum/atom:array_loc");
    for (i = 0; i < omega_n; i++)
      for (j= 0; j < nvalues; j++)
        array_loc[i][j]=0.0;

    for (i = 0; i < omega_n; i++)
      for (j= 0; j < nvalues; j++)
        for (n=0; n < atom->nlocal; n++)
          array_loc[i][j] += v_omega_r[n][j][i]*v_omega_r[n][j][i] +
            v_omega_i[n][j][i]*v_omega_i[n][j][i];

    MPI_Allreduce(&array_loc[0][0],&array[0][0],
        omega_n*nvalues, MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < omega_n; i++)
      for (j= 0; j < nvalues; j++)
        array[i][j] = nevery*array[i][j]/count*update->dt/atom->natoms;
        //array[i][j] = array[i][j]*nevery*update->dt/count/atom->natoms;
    memory->destroy(array_loc);

    if (fp && me == 0) {
      clearerr(fp);
      if (overwrite) fseek(fp, filepos, SEEK_SET);
      //fprintf(fp,BIGINT_FORMAT "\n", update->ntimestep);
      for (i = 0; i < omega_n; i++ ) {
        fprintf(fp, "%g", omega_vec[i]);
        for (j = 0; j < nvalues; j++ ) {
          fprintf(fp, " %g", array[i][j]);
        }
        fprintf(fp, "\n");
      }
      if (ferror(fp))
        error->one(FLERR,"Error writing out spectral data");

      fflush(fp);

      if (overwrite) {
        long fileend = ftell(fp);
        if (fileend > 0) ftruncate(fileno(fp),fileend);
      }
    }



  }
  nvalid = ntimestep+nevery;
  modify->addstep_compute(nvalid);

}

/* ----------------------------------------------------------------------
   memory usage 
------------------------------------------------------------------------- */

double FixSpectrumAtom::memory_usage()
{
  double bytes;
  //bytes = omega_n* (nvalues) * sizeof(double);
  bytes = 2*omega_n*atom->nmax*nvalues * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixSpectrumAtom::grow_arrays(int nmax)
{
  v_omega_r = memory->grow(v_omega_r,nmax,nvalues,omega_n,
      "fix_spectrum/atom:v_omega_r");
  v_omega_i = memory->grow(v_omega_i,nmax,nvalues,omega_n,
      "fix_spectrum/atom:v_omega_i");
}
/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */
void FixSpectrumAtom::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int n = 0; n < nvalues; n++)
    for (int k = 0; k < omega_n; k++) {
      v_omega_r[j][n][k] = v_omega_r[i][n][k];
      v_omega_i[j][n][k] = v_omega_i[i][n][k];
    }
}

int FixSpectrumAtom::pack_exchange(int i, double *buf)
{
  int m = 0;
  for (int j = 0; j < nvalues; j++)
    for (int k = 0; k < omega_n; k++) {
      buf[m++] = v_omega_r[i][j][k];
      buf[m++] = v_omega_i[i][j][k];
    }
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixSpectrumAtom::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int j = 0; j < nvalues; j++)
    for (int k = 0; k < omega_n; k++)  {
      v_omega_r[nlocal][j][k] = buf[m++];
      v_omega_i[nlocal][j][k] = buf[m++];
    }
  return m;
}

double FixSpectrumAtom::compute_array(int i, int j)
{
  if (i<=1)
    return omega_vec[j];
  else
    return array[i-1][j];
}
