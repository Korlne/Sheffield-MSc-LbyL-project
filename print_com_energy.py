import ROOT
root_file = "/home/jtong/lbyl/alp_binned/user.slawlor.Lbylntuples.periodAllYear2.DAOD_HION4.grp23_v01_p6774_198765_Smc5000_Sppc2500.root"

f = ROOT.TFile.Open(root_file)
f.ls()
