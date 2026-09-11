#include "gadget_params.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

// Phase 5 ticket 01 (PLAN.md) -- see gadget_params.h for the design rationale. This file's
// structure deliberately mirrors read_parameter_file()'s own tag[]/addr[]/id[] table
// (begrun.c:280-752) so it stays easy to audit against the source: one entry per tag, a type tag,
// and a pointer to the field it fills.

namespace {

enum class TagType { DOUBLE, STRING, INT };

struct TagEntry
{
  const char *name;
  TagType     type;
  void       *addr;
};

// SEC_PER_MEGAYEAR / GRAVITY / HUBBLE: physical constants Gadget-2 hardcodes in allvars.h (61,
// 75, 79) and uses in set_units() (begrun.c:149-191) -- copied verbatim, these are not derived
// from anything else.
constexpr double SEC_PER_MEGAYEAR = 3.155e13;
constexpr double GRAVITY_CGS      = 6.672e-8;
constexpr double HUBBLE_PER_SEC   = 3.2407789e-18; // H0 in h/sec

// Builds the tag table against a live GadgetParams instance. Order matches
// PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 1.5's registration-order table (itself begrun.c:329-587) --
// kept in the same order purely so a side-by-side diff against the spec/source stays easy, order
// has no functional significance to the parser itself.
std::vector<TagEntry> buildTagTable(GadgetParams &p)
{
  return {
    { "InitCondFile",              TagType::STRING, &p.InitCondFile },
    { "OutputDir",                 TagType::STRING, &p.OutputDir },
    { "SnapshotFileBase",          TagType::STRING, &p.SnapshotFileBase },
    { "EnergyFile",                TagType::STRING, &p.EnergyFile },
    { "CpuFile",                   TagType::STRING, &p.CpuFile },
    { "InfoFile",                  TagType::STRING, &p.InfoFile },
    { "TimingsFile",               TagType::STRING, &p.TimingsFile },
    { "RestartFile",               TagType::STRING, &p.RestartFile },
    { "ResubmitCommand",           TagType::STRING, &p.ResubmitCommand },
    { "OutputListFilename",        TagType::STRING, &p.OutputListFilename },
    { "OutputListOn",              TagType::INT,    &p.OutputListOn },
    { "Omega0",                    TagType::DOUBLE, &p.Omega0 },
    { "OmegaBaryon",               TagType::DOUBLE, &p.OmegaBaryon },
    { "OmegaLambda",               TagType::DOUBLE, &p.OmegaLambda },
    { "HubbleParam",               TagType::DOUBLE, &p.HubbleParam },
    { "BoxSize",                   TagType::DOUBLE, &p.BoxSize },
    { "PeriodicBoundariesOn",      TagType::INT,    &p.PeriodicBoundariesOn },
    { "TimeOfFirstSnapshot",       TagType::DOUBLE, &p.TimeOfFirstSnapshot },
    { "CpuTimeBetRestartFile",     TagType::DOUBLE, &p.CpuTimeBetRestartFile },
    { "TimeBetStatistics",         TagType::DOUBLE, &p.TimeBetStatistics },
    { "TimeBegin",                 TagType::DOUBLE, &p.TimeBegin },
    { "TimeMax",                   TagType::DOUBLE, &p.TimeMax },
    { "TimeBetSnapshot",           TagType::DOUBLE, &p.TimeBetSnapshot },
    { "UnitVelocity_in_cm_per_s",  TagType::DOUBLE, &p.UnitVelocity_in_cm_per_s },
    { "UnitLength_in_cm",          TagType::DOUBLE, &p.UnitLength_in_cm },
    { "UnitMass_in_g",             TagType::DOUBLE, &p.UnitMass_in_g },
    { "TreeDomainUpdateFrequency", TagType::DOUBLE, &p.TreeDomainUpdateFrequency },
    { "ErrTolIntAccuracy",         TagType::DOUBLE, &p.ErrTolIntAccuracy },
    { "ErrTolTheta",               TagType::DOUBLE, &p.ErrTolTheta },
    { "ErrTolForceAcc",            TagType::DOUBLE, &p.ErrTolForceAcc },
    { "MinGasHsmlFractional",      TagType::DOUBLE, &p.MinGasHsmlFractional },
    { "MaxSizeTimestep",           TagType::DOUBLE, &p.MaxSizeTimestep },
    { "MinSizeTimestep",           TagType::DOUBLE, &p.MinSizeTimestep },
    { "MaxRMSDisplacementFac",     TagType::DOUBLE, &p.MaxRMSDisplacementFac },
    { "ArtBulkViscConst",          TagType::DOUBLE, &p.ArtBulkViscConst },
    { "CourantFac",                TagType::DOUBLE, &p.CourantFac },
    { "DesNumNgb",                 TagType::DOUBLE, &p.DesNumNgb },
    { "MaxNumNgbDeviation",        TagType::DOUBLE, &p.MaxNumNgbDeviation },
    { "ComovingIntegrationOn",     TagType::INT,    &p.ComovingIntegrationOn },
    { "ICFormat",                  TagType::INT,    &p.ICFormat },
    { "SnapFormat",                TagType::INT,    &p.SnapFormat },
    { "NumFilesPerSnapshot",       TagType::INT,    &p.NumFilesPerSnapshot },
    { "NumFilesWrittenInParallel", TagType::INT,    &p.NumFilesWrittenInParallel },
    { "ResubmitOn",                TagType::INT,    &p.ResubmitOn },
    { "TypeOfTimestepCriterion",   TagType::INT,    &p.TypeOfTimestepCriterion },
    { "TypeOfOpeningCriterion",    TagType::INT,    &p.TypeOfOpeningCriterion },
    { "TimeLimitCPU",              TagType::DOUBLE, &p.TimeLimitCPU },
    { "SofteningHalo",             TagType::DOUBLE, &p.SofteningHalo },
    { "SofteningDisk",             TagType::DOUBLE, &p.SofteningDisk },
    { "SofteningBulge",            TagType::DOUBLE, &p.SofteningBulge },
    { "SofteningGas",              TagType::DOUBLE, &p.SofteningGas },
    { "SofteningStars",            TagType::DOUBLE, &p.SofteningStars },
    { "SofteningBndry",            TagType::DOUBLE, &p.SofteningBndry },
    { "SofteningHaloMaxPhys",      TagType::DOUBLE, &p.SofteningHaloMaxPhys },
    { "SofteningDiskMaxPhys",      TagType::DOUBLE, &p.SofteningDiskMaxPhys },
    { "SofteningBulgeMaxPhys",     TagType::DOUBLE, &p.SofteningBulgeMaxPhys },
    { "SofteningGasMaxPhys",       TagType::DOUBLE, &p.SofteningGasMaxPhys },
    { "SofteningStarsMaxPhys",     TagType::DOUBLE, &p.SofteningStarsMaxPhys },
    { "SofteningBndryMaxPhys",     TagType::DOUBLE, &p.SofteningBndryMaxPhys },
    { "BufferSize",                TagType::INT,    &p.BufferSize },
    { "PartAllocFactor",           TagType::DOUBLE, &p.PartAllocFactor },
    { "TreeAllocFactor",           TagType::DOUBLE, &p.TreeAllocFactor },
    { "GravityConstantInternal",   TagType::DOUBLE, &p.GravityConstantInternal },
    { "InitGasTemp",               TagType::DOUBLE, &p.InitGasTemp },
    { "MinGasTemp",                TagType::DOUBLE, &p.MinGasTemp },
  };
}

// Splits a line into at most 3 whitespace-separated tokens, matching begrun.c:603's
// `sscanf(buf, "%s%s%s", buf1, buf2, buf3)` -- anything past the 3rd token is simply never
// captured, exactly like the source (spec Sec 1.1). Returns the number of tokens actually found
// (0, 1, 2, or 3+; 3+ collapses to 3 since the rest is discarded either way).
int tokenize3(const std::string &line, std::string &tok1, std::string &tok2, std::string &tok3)
{
  std::istringstream iss(line);
  std::vector<std::string> toks;
  std::string t;
  while (toks.size() < 3 && (iss >> t))
    toks.push_back(t);
  // Peek whether a 4th token exists, purely to report an accurate count -- content is discarded
  // either way, matching source behavior of just not capturing it.
  int extra = 0;
  if (toks.size() == 3 && (iss >> t))
    extra = 1;
  if (toks.size() > 0) tok1 = toks[0];
  if (toks.size() > 1) tok2 = toks[1];
  if (toks.size() > 2) tok3 = toks[2];
  return (int)toks.size() + extra;
}

} // namespace

void GadgetParams::deriveUnits()
{
  // set_units(), begrun.c:149-191 -- formulas copied exactly, including operation order, so the
  // resulting G/Hubble are bit-for-bit the same shape as stock Gadget-2's for the same inputs.
  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
  UnitTime_in_Megayears = UnitTime_in_s / SEC_PER_MEGAYEAR;

  if (GravityConstantInternal == 0.0)
  {
    G = GRAVITY_CGS / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm)
        * UnitMass_in_g * (UnitTime_in_s * UnitTime_in_s);
  }
  else
  {
    G = GravityConstantInternal;
  }

  UnitDensity_in_cgs = UnitMass_in_g / (UnitLength_in_cm * UnitLength_in_cm * UnitLength_in_cm);

  Hubble = HUBBLE_PER_SEC * UnitTime_in_s;
}

bool gadget_params_parse(const std::string &filename, bool builtWithPeriodic, GadgetParams &out,
                          std::vector<std::string> &errors)
{
  errors.clear();
  out = GadgetParams();

  std::ifstream fd(filename);
  if (!fd.is_open())
  {
    errors.push_back("Cannot open parameter file '" + filename + "'");
    return false;
  }

  std::vector<TagEntry> table = buildTagTable(out);
  std::map<std::string, size_t> byName;
  for (size_t i = 0; i < table.size(); ++i)
    byName[table[i].name] = i;
  std::vector<bool> seen(table.size(), false);

  std::string line;
  int lineNo = 0;
  while (std::getline(fd, line))
  {
    ++lineNo;
    std::string tok1, tok2, tok3;
    int n = tokenize3(line, tok1, tok2, tok3);

    // begrun.c:604 -- fewer than 2 tokens on a line is treated as blank, not an error.
    if (n < 2)
      continue;

    // begrun.c:606 -- a line is a comment only if its FIRST token starts with '%'. A trailing
    // remark after the value (e.g. "; here in seconds") is not special-cased at all; it just
    // lands in the discarded 3rd+ token, matching spec Sec 1.1 exactly.
    if (!tok1.empty() && tok1[0] == '%')
      continue;

    auto it = byName.find(tok1);
    if (it == byName.end() || seen[it->second])
    {
      // begrun.c:637-639 -- both "no such tag" and "tag already matched once" share this single
      // error message in the source; kept as one combined case here too.
      errors.push_back(filename + ":" + std::to_string(lineNo) +
                        ": tag '" + tok1 + "' not allowed or multiple defined");
      continue;
    }

    const TagEntry &e = table[it->second];
    switch (e.type)
    {
      case TagType::DOUBLE:
        *static_cast<double *>(e.addr) = std::strtod(tok2.c_str(), nullptr);
        break;
      case TagType::INT:
        *static_cast<int *>(e.addr) = std::atoi(tok2.c_str());
        break;
      case TagType::STRING:
        *static_cast<std::string *>(e.addr) = tok2;
        break;
    }
    seen[it->second] = true;
  }

  // begrun.c:662-670 -- any tag never matched by EOF is a hard error. Every one of the 65 tags is
  // required unconditionally (PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 0) -- there is no default-value
  // mechanism to fall back to.
  for (size_t i = 0; i < table.size(); ++i)
  {
    if (!seen[i])
      errors.push_back(filename + ": missing required tag '" + std::string(table[i].name) +
                        "'");
  }

  if (!errors.empty())
    return false;

  // Post-parse validation gates (begrun.c Sec 1.2) that remain meaningful for this single-
  // process, TypeOfTimestepCriterion==0-only port.
  if (out.NumFilesWrittenInParallel != 1)
  {
    errors.push_back("NumFilesWrittenInParallel must be 1 for this single-process port (got " +
                      std::to_string(out.NumFilesWrittenInParallel) + ")");
  }
  if (out.TypeOfTimestepCriterion != 0)
  {
    errors.push_back("TypeOfTimestepCriterion: only 0 is supported (matches stock Gadget-2, "
                      "begrun.c:727-735), got " + std::to_string(out.TypeOfTimestepCriterion));
  }
  {
    int expected = builtWithPeriodic ? 1 : 0;
    if (out.PeriodicBoundariesOn != expected)
    {
      errors.push_back("PeriodicBoundariesOn=" + std::to_string(out.PeriodicBoundariesOn) +
                        " does not match this build's periodicity mode (built " +
                        (builtWithPeriodic ? "WITH" : "WITHOUT") + " PERIODIC)");
    }
  }

  if (!errors.empty())
    return false;

  out.deriveUnits();
  return true;
}
