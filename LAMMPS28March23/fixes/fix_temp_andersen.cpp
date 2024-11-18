// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_temp_andersen.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "utils.h"
#include "variable.h"
#include "random_mars.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
/* ---------------------------------------------------------------------- */

FixTempAndersen::FixTempAndersen(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  id_temp(nullptr), tflag(0)
{
  if (narg != 7)
    error->all(FLERR,"Illegal fix {} command: expected atleast 6 arguments but found {}", style, narg);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR, "Invalid fix temp/andersen every argument: {}", nevery);
  // Andersen thermostat should be applied every step

  restart_global = 1;
  dynamic_group_allow = 1;
  scalar_flag = 1;
  extscalar = 1;
  ecouple_flag = 1;
  global_freq = nevery;

  Tbath = utils::numeric(FLERR,arg[4],false,lmp);

  Nfraction = utils::numeric(FLERR,arg[5],false,lmp); // should be between 0 and 1
  if (Nfraction > 1 || Nfraction < 0) error->all(FLERR, "Invalid fix temp/andersen Nfarction argument: {}", Nfraction);

  seedfix = utils::inumeric(FLERR,arg[6],false,lmp);
  if (seedfix <= 0 ) error->all(FLERR,"Random seed must be a positive number");

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp",id_temp,group->names[igroup]));
  tflag = 1;

  energy = 0;
  
  // random number generator

  random = new RanMars(lmp,seedfix + comm->me);
}

/* ---------------------------------------------------------------------- */

FixTempAndersen::~FixTempAndersen()
{
  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete[] id_temp;
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixTempAndersen::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::init()
{

  temperature = modify->get_compute_by_id(id_temp);
  if (!temperature)
    error->all(FLERR,"Temperature compute ID {} for fix {} does not exist", id_temp, style);

  if (modify->check_rigid_group_overlap(groupbit))
    error->warning(FLERR,"Cannot thermostat atoms in rigid bodies with fix {}", style);

  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;
}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::end_of_step()
{
  double t_current = temperature->compute_scalar();
  double tdof = temperature->dof;
  double factor, difftest, theta_i, theta_f;

  // there is nothing to do, if there are no degrees of freedom

  if (tdof < 1) return;

  if (t_current == 0.0)
    error->all(FLERR, "Computed current temperature for fix temp/andersen must not be 0.0");

  double **v = atom->v;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  theta_i = force->boltz*Tbath/force->mvv2e;

  energy += 0.5*t_current*force->boltz*temperature->dof;

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
  	theta_f = theta_i/(mass[type[i]]);	
 	factor = sqrt(theta_f);
	difftest = random->uniform();
	if(difftest <= Nfraction)
	{
		v[i][0]=random->gaussian(0,factor);
		v[i][1]=random->gaussian(0,factor);
		v[i][2]=random->gaussian(0,factor);
	}
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
	      theta_f = theta_i/(mass[type[i]]);
	      factor = sqrt(theta_f);
	      difftest = random->uniform();
	      if(difftest <= Nfraction)
	      {
		      temperature->remove_bias(i,v[i]);
		      v[i][0] = random->gaussian(0,factor);
		      v[i][1] = random->gaussian(0,factor);
		      v[i][2] = random->gaussian(0,factor);
		      temperature->restore_bias(i,v[i]);
	      }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixTempAndersen::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify", error);
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);

    temperature = modify->get_compute_by_id(id_temp);
    if (!temperature)
      error->all(FLERR,"Could not find fix_modify temperature compute {}", id_temp);

    if (temperature->tempflag == 0)
      error->all(FLERR, "Fix_modify temperature compute {} does not compute temperature", id_temp);
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR, "Group for fix_modify temp != fix group: {} vs {}",
                     group->names[igroup], group->names[temperature->igroup]);
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::reset_target(double t_new)
{
  Tbath = t_new;
}

/* ---------------------------------------------------------------------- */

double FixTempAndersen::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixTempAndersen::write_restart(FILE *fp)
{
  int n = 0;
  double list[1];
  list[n++] = energy;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixTempAndersen::restart(char *buf)
{
  auto list = (double *) buf;

  energy = list[0];
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixTempAndersen::extract(const char *str, int &dim)
{
  dim=0;
  if (strcmp(str,"Tbath") == 0) {
    return &Tbath;
  }
  return nullptr;
}
