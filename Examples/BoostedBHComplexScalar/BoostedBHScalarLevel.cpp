/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "BoostedBHScalarLevel.hpp"
#include "AMRReductions.hpp"
#include "BoxLoops.hpp"
#include "ComputePack.hpp"
#include "NanCheck.hpp"
#include "SetValue.hpp"
#include "SmallDataIO.hpp"

// For RHS update
#include "BoostedBH.hpp"
#include "MatterEvolution.hpp"

// For tag cells
#include "FixedGridsTaggingCriterion.hpp"

// Problem specific includes
#include "ComplexScalarField.hpp"
#include "ComplexScalarPotential.hpp"
#include "EnergyConservation.hpp"
#include "ExcisionDiagnostics.hpp"
#include "ExcisionEvolution.hpp"
#include "FluxExtraction.hpp"
#include "InitialScalarData.hpp"
#include "LinearMomConservation.hpp"

// Initial data for field and metric variables
void BoostedBHScalarLevel::initialData()
{
    CH_TIME("BoostedBHScalarLevel::initialData");
    if (m_verbosity)
        pout() << "BoostedBHScalarLevel::initialData " << m_level << endl;

    // First set everything to zero, then set the value of the conformal factor
    // This is just for the diagnostics
    SetValue set_zero(0.0);
    BoostedBH boosted_bh(m_p.bg_params, m_dx); // just calculates chi
    auto compute_pack = make_compute_pack(set_zero, boosted_bh);
    BoxLoops::loop(compute_pack, m_state_diagnostics, m_state_diagnostics,
                   SKIP_GHOST_CELLS);

    // Now set the actual evolution variables
    InitialScalarData initial_sf(m_p.initial_params);
    BoxLoops::loop(initial_sf, m_state_new, m_state_new, FILL_GHOST_CELLS);

    // excise evolution vars within horizon, turn off simd vectorisation
    BoxLoops::loop(ExcisionEvolution<ScalarFieldWithPotential, BoostedBH>(
                       m_dx, m_p.center, boosted_bh),
                   m_state_new, m_state_new, SKIP_GHOST_CELLS, disable_simd());
}

void BoostedBHScalarLevel::specificPostTimeStep()
{
    // Check for nans on every level
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new, SKIP_GHOST_CELLS,
                       disable_simd());

    // At any level, but after the timestep on the minimum extraction level
    int min_level = 0.;
    bool calculate_diagnostics = at_level_timestep_multiple(min_level);
    if (calculate_diagnostics)
    {
        fillAllGhosts();
        ComplexScalarPotential potential(m_p.initial_params);
        ScalarFieldWithPotential scalar_field(potential);
        BoostedBH boosted_bh(m_p.bg_params, m_dx);
        EnergyConservation<ScalarFieldWithPotential, BoostedBH> energies(
            scalar_field, boosted_bh, m_dx, m_p.center);
        int direction = 0; // we want the x direction for the momentum
        LinearMomConservation<ScalarFieldWithPotential, BoostedBH>
            linear_momenta(scalar_field, boosted_bh, direction, m_dx,
                           m_p.center);
        BoxLoops::loop(make_compute_pack(energies, linear_momenta), m_state_new,
                       m_state_diagnostics, SKIP_GHOST_CELLS);

        // excise within/outside specified radii, no simd
        if (m_p.activate_extraction == 1)
        {
            BoxLoops::loop(
                ExcisionDiagnostics<ScalarFieldWithPotential, BoostedBH>(
                    m_dx, m_p.center, boosted_bh, m_p.inner_r, m_p.outer_r),
                m_state_diagnostics, m_state_diagnostics, SKIP_GHOST_CELLS,
                disable_simd());
        }
    }

    // write out the integral after each timestep on the min_level
    if (m_p.activate_extraction == 1)
    {
        min_level = m_p.extraction_params.min_extraction_level();
        if (m_level == min_level)
        {
            bool first_step = (m_time == m_dt);
            // integrate the densities and write to a file
            AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
            double rhoEnergy_sum = amr_reductions.sum(c_rhoEnergy);
            double rhoLinMom_sum = amr_reductions.sum(c_rhoLinMom);
            double sourceLinMom_sum = amr_reductions.sum(c_sourceLinMom);

            SmallDataIO integral_file(m_p.data_path + "EnergyIntegrals", m_dt,
                                      m_time, m_restart_time,
                                      SmallDataIO::APPEND, first_step);
            // remove any duplicate data if this is post restart
            integral_file.remove_duplicate_time_data();

            std::vector<double> data_for_writing = {
                rhoEnergy_sum, rhoLinMom_sum, sourceLinMom_sum};

            // write data
            if (first_step)
            {
                integral_file.write_header_line({"Energy density.",
                                                 "Lin. Mom. density",
                                                 "Lin. Mom. source"});
            }
            integral_file.write_time_data_line(data_for_writing);

            // Now refresh the interpolator and do the interpolation
            // only fill the actual ghost cells needed to save time
            bool fill_ghosts = false;
            m_gr_amr.m_interpolator->refresh(fill_ghosts);
            m_gr_amr.fill_multilevel_ghosts(
                VariableType::diagnostic, Interval(c_fluxEnergy, c_fluxLinMom));
            FluxExtraction my_extraction(m_p.extraction_params, m_dt, m_time,
                                         m_restart_time);
            my_extraction.execute_query(m_gr_amr.m_interpolator, m_p.data_path);
        }
    }
}

// Things to do in RHS update, at each RK4 step
void BoostedBHScalarLevel::specificEvalRHS(GRLevelData &a_soln,
                                           GRLevelData &a_rhs,
                                           const double a_time)
{
    // Calculate right hand side with matter_t = ScalarField
    // and background_t = BoostedBH
    ComplexScalarPotential potential(m_p.initial_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoostedBH boosted_bh(m_p.bg_params, m_dx);
    MatterEvolution<ScalarFieldWithPotential, BoostedBH> my_evolution(
        scalar_field, boosted_bh, m_p.sigma, m_dx, m_p.center);
    BoxLoops::loop(my_evolution, a_soln, a_rhs, SKIP_GHOST_CELLS);

    // Do excision within horizon, don't use vectorisation
    BoxLoops::loop(ExcisionEvolution<ScalarFieldWithPotential, BoostedBH>(
                       m_dx, m_p.center, boosted_bh),
                   a_soln, a_rhs, SKIP_GHOST_CELLS, disable_simd());
}

// Note that for the fixed grids this only happens on the initial timestep
void BoostedBHScalarLevel::computeTaggingCriterion(
    FArrayBox &tagging_criterion, const FArrayBox &current_state)
{
    BoxLoops::loop(FixedGridsTaggingCriterion(m_dx, m_level, m_p.L, m_p.center),
                   current_state, tagging_criterion);
}
