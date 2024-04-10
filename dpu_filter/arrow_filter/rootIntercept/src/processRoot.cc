// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TTreePerfStats.h"
#include <iostream>
#include <math.h>
#include "TROOT.h"
#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fstream>
#include "TEnv.h"
#include "TFileCacheRead.h"
#include "TMemFile.h"
#include "TBufferFile.h"


const char *branchNames[] = {"nElectron", "Electron_pt", "Electron_eta", "Electron_phi", "Electron_deltaEtaSC",
                             "Electron_dz", "Electron_dxy", "Electron_cutBased", "nMuon", "Muon_pt", "Muon_eta",
                             "Muon_phi", "Muon_tightId", "Muon_pfRelIso04_all", "nJet", "Jet_pt", "Jet_eta",
                             "Jet_phi", "nFatJet", "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass",
                             "FatJet_msoftdrop", "FatJet_particleNetMD_Xbb", "FatJet_particleNetMD_QCD", "MET_pt"};
const int nBranches = sizeof(branchNames) / sizeof(branchNames[0]);


float Phi_mpi_pi(float x)
{
    if (isnan(x))
    {
        std::cout << "Phi_mpi_pi(float)function called with NaN" << std::endl;
        return x;
    }
    while (x >= M_PI) x -= 2*M_PI;
    while (x < -M_PI) x += 2*M_PI;
    return x;
}

float DeltaR(float eta1, float phi1, float eta2, float phi2)
{
    float deta = eta1 - eta2;
    float dphi = Phi_mpi_pi(phi1 - phi2);
    return sqrt(deta * deta + dphi * dphi);
}

extern "C" {

    unsigned char* processRoot(const unsigned char* data_buffer, size_t data_size, size_t& filtered_size) {

     auto start = std::chrono::system_clock::now();
        using std::chrono::high_resolution_clock;
    
auto start_m = std::chrono::system_clock::now();
        // Initialize ROOT's TMemFile from the input buffer
        TMemFile memFile("memfile.root", const_cast<char*>(reinterpret_cast<const char*>(data_buffer)), static_cast<Long64_t>(data_size), "READ");



        if (!memFile.IsOpen()) {
            std::cerr << "Failed to open TMemFile" << std::endl;
            return nullptr;
        }

   
    auto stop_m = std::chrono::system_clock::now();
    auto total_m = std::chrono::duration_cast<std::chrono::milliseconds>(stop_m - start_m);


    std::cout  <<  "  memfile " <<  total_m.count() << " milliseconds\n";

       

auto start_e = std::chrono::system_clock::now();
        // Access the TTree object from the TMemFile
        TTree* tree = static_cast<TTree*>(memFile.Get("Events"));


        if (!tree) {
            std::cerr << "Failed to get TTree from TMemFile" << std::endl;
            return nullptr;
        }

    auto stop_e = std::chrono::system_clock::now();
    auto total_e = std::chrono::duration_cast<std::chrono::milliseconds>(stop_e - start_e);


    std::cout  <<  "  GetEvent " <<  total_e.count() << " milliseconds\n";

        auto start_i = std::chrono::system_clock::now();

        bool TurnOnPartialBranches = true;

    if (TurnOnPartialBranches)
    {
        // Turning off the status of the unused branches to not load them every time
        // This speed things up in terms of skimming however, it also means none of the other branches are copied
        tree->SetBranchStatus("*", false);
        tree->SetBranchStatus("nElectron*", true); //(UInt 32 bit unsigned int) trivial
        tree->SetBranchStatus("nMuon*", true); //(UInt 32 bit unsigned int) trivial
        tree->SetBranchStatus("nFatJet*", true); //(UInt 32 bit unsigned int) trivial
        tree->SetBranchStatus("nJet*", true); //(UInt 32 bit unsigned int) trivial

        tree->SetBranchStatus("Electron*", true); //13 sec
        tree->SetBranchStatus("Muon*", true); //10sec
        tree->SetBranchStatus("Jet*", true); //23sec
        tree->SetBranchStatus("Tau*", true); //7sec
        tree->SetBranchStatus("GenPart*", true); //20sec
        tree->SetBranchStatus("Generator*", true); //1 sec
        tree->SetBranchStatus("FatJet*", true); //6 sec
        tree->SetBranchStatus("MET*", true); // 2 sec only floats stopped at 34sec

        tree->SetBranchStatus("event*", true); // 1sec very fast ULong64_t
        tree->SetBranchStatus("run*", true); //UInt_t	very fast less than second
        tree->SetBranchStatus("luminosityBlock*", true); // UInt_t very fast
        tree->SetBranchStatus("genWeight*", true); // Float_t very fast
        tree->SetBranchStatus("btagWeight*", true); //Float_t very fast
        tree->SetBranchStatus("LHE*", true); // 6.9 sec ?? 6 sec minus in decompression
        tree->SetBranchStatus("LHEPdfWeight*", true);
        tree->SetBranchStatus("*Weight*", true); //Float_t 13seconds many branches, 9 sec minus in decompression (oor oor gazar)
        tree->SetBranchStatus("Flag*", true); //Bool_t 1second, many branches
        tree->SetBranchStatus("SubJet*", true); //2seconds Float_t, Int_t, UChar_t, UInt_t (neg gazar buugnursun)
        tree->SetBranchStatus("HLT_IsoMu*", true); //boolean main bottleneck 100sec
        tree->SetBranchStatus("HLT_Ele27_WPTight_Gsf", true); //boolean main bottleneck 100sec
        tree->SetBranchStatus("HLT_Ele32_WPTight_Gsf", true); //boolean main bottleneck 100sec
        tree->SetBranchStatus("Pileup*", true); // Float_t, Int_t 0.6 sec fast*/
    }

    //TTreePerfStats *ps= new TTreePerfStats("ioperf", tree);

    // Initialize an output TMemFile
    TMemFile outputFile("output.root", "RECREATE");
    // Clone the tree to the output TMemFile
    outputFile.cd(); // Make sure we are in the outputFile's directory
    TTree* outputTree = tree->CloneTree(0); // Clone structure, no entries yet

    int finalevents = 0; //final event number
   

    // Defining "column" variable memories
    // --- Electron columns
    unsigned int nElectron;
    float Electron_pt[90];
    float Electron_eta[90];
    float Electron_phi[90];
    float Electron_deltaEtaSC[90];
    float Electron_dz[90];
    float Electron_dxy[90];
    int Electron_cutBased[90];
    // --- Muon columns
    unsigned int nMuon;
    bool Muon_tightId[90];
    float Muon_pfRelIso04_all[90];
    float Muon_pt[90];
    float Muon_eta[90];
    float Muon_phi[90];
    // --- Jet columns
    unsigned int nJet;
    float Jet_pt[250];
    float Jet_eta[250];
    float Jet_phi[250];
    // --- FatJet columns
    unsigned int nFatJet;
    float FatJet_pt[18];
    float FatJet_eta[18];
    float FatJet_phi[18];
    float FatJet_mass[18];
    float FatJet_msoftdrop[18];
    float FatJet_particleNetMD_Xbb[18];
    float FatJet_particleNetMD_QCD[18];
    // --- MET columns
    float MET_pt;

    // Setting the address of variable memories to the TTree object's address
    // This way, whenever ''tree->GetEntry(ith-entry)'' is called the column variables
    // get assigned with the values for the ith-event.
    // --- Electron columns
    tree->SetBranchAddress("nElectron", &nElectron);
    tree->SetBranchAddress("Electron_pt", &Electron_pt);
    tree->SetBranchAddress("Electron_eta", &Electron_eta);
    tree->SetBranchAddress("Electron_phi", &Electron_phi);
    tree->SetBranchAddress("Electron_deltaEtaSC", &Electron_deltaEtaSC);
    tree->SetBranchAddress("Electron_dz", &Electron_dz);
    tree->SetBranchAddress("Electron_dxy", &Electron_dxy);
    tree->SetBranchAddress("Electron_cutBased", &Electron_cutBased);
    // --- Muon columns
    tree->SetBranchAddress("nMuon", &nMuon);
    tree->SetBranchAddress("Muon_pt", &Muon_pt);
    tree->SetBranchAddress("Muon_eta", &Muon_eta);
    tree->SetBranchAddress("Muon_phi", &Muon_phi);
    tree->SetBranchAddress("Muon_tightId", &Muon_tightId);
    tree->SetBranchAddress("Muon_pfRelIso04_all", &Muon_pfRelIso04_all);
    // --- Jet columns
    tree->SetBranchAddress("nJet", &nJet);
    tree->SetBranchAddress("Jet_pt", &Jet_pt);
    tree->SetBranchAddress("Jet_eta", &Jet_eta);
    tree->SetBranchAddress("Jet_phi", &Jet_phi);
    // --- FatJet columns
    tree->SetBranchAddress("nFatJet", &nFatJet);
    tree->SetBranchAddress("FatJet_pt", &FatJet_pt);
    tree->SetBranchAddress("FatJet_eta", &FatJet_eta);
    tree->SetBranchAddress("FatJet_phi", &FatJet_phi);
    tree->SetBranchAddress("FatJet_mass", &FatJet_mass);
    tree->SetBranchAddress("FatJet_msoftdrop", &FatJet_msoftdrop);
    tree->SetBranchAddress("FatJet_particleNetMD_Xbb", &FatJet_particleNetMD_Xbb);
    tree->SetBranchAddress("FatJet_particleNetMD_QCD", &FatJet_particleNetMD_QCD);
    // --- MET columns
    tree->SetBranchAddress("MET_pt", &MET_pt);

    

    TBranch *branches[nBranches];
    for (int i = 0; i < nBranches; ++i) {
        branches[i] = tree->GetBranch(branchNames[i]);
        //tree->AddBranchToCache(branches[i], true);
    }

    auto stop_i = std::chrono::system_clock::now();
    auto total_i = std::chrono::duration_cast<std::chrono::milliseconds>(stop_i - start_i);


    std::cout  <<  "  Init " <<  total_i.count() << " milliseconds\n";
    //tree->Print("clusters");

    auto start_f = std::chrono::system_clock::now();

    long totread = 0;
    for (unsigned int ientry = 0; ientry < tree->GetEntries(); ++ientry)
    //for (unsigned int ientry = 0; ientry < 10000; ++ientry)
    {
        
        for (int b = 0; b < nBranches; ++b) {
            branches[b]->GetEntry(ientry);
        }
            
        
        /*if (ientry % 10000 == 0)
            std::cout <<  " ientry: " << ientry <<  std::endl;*/

        // With the following function call, the "column" variables are filled
        //tree->GetEntry(ientry);
    
        // Per event variables that we want to compute (i.e. new "column" variables to compute)
        int nVetoLepton = 0;
        int nTightLepton = 0;
        int nGoodJet = 0;
        int nGoodFatJet = 0;

        std::vector<float> TightLepton_pt;
        std::vector<float> TightLepton_eta;
        std::vector<float> TightLepton_phi;

        float MaxHbbScore = -999;
        float HbbFatJet_pt = 0;

        // Selection for Electrons
        for (unsigned int iElec = 0; iElec < nElectron; ++iElec)
        {
            // Veto lepton selections
            if (Electron_pt[iElec] <= 10)
                continue;
            if (Electron_cutBased[iElec] < 1)
                continue;

            // Count the number of veto leptons
            nVetoLepton++;

            // Tight lepton selections
            if (Electron_pt[iElec] <= 35)
                continue;
            if (Electron_cutBased[iElec] < 3)
                continue;
            if (fabs(Electron_eta[iElec] + Electron_deltaEtaSC[iElec]) >= 2.5)
                continue;
            if (fabs(Electron_eta[iElec] + Electron_deltaEtaSC[iElec]) >= 1.479)
            {
                if (fabs(Electron_dz[iElec]) >= 0.2)
                    continue;
                if (fabs(Electron_dxy[iElec]) >= 0.1)
                    continue;
            }
            else
            {
                if (fabs(Electron_dz[iElec]) >= 0.1)
                    continue;
                if (fabs(Electron_dxy[iElec]) >= 0.05)
                    continue;
            }

            TightLepton_pt.push_back(Electron_pt[iElec]);
            TightLepton_eta.push_back(Electron_eta[iElec]);
            TightLepton_phi.push_back(Electron_phi[iElec]);

            // Count the number of tight leptons
            nTightLepton++;
        }

        // Selection for Muons
        for (unsigned int iMuon = 0; iMuon < nElectron; ++iMuon)
        {
            // Veto lepton selections
            if (not Muon_tightId[iMuon])
                continue;
            if (Muon_pfRelIso04_all[iMuon] >= 0.4)
                continue;
            if (Muon_pt[iMuon] <= 10)
                continue;

            // Count the number of veto leptons
            nVetoLepton++;

            // Tight lepton selections
            if (Muon_pfRelIso04_all[iMuon] >= 0.15)
                continue;
            if (Muon_pt[iMuon] <= 26)
                continue;
            if (fabs(Muon_eta[iMuon]) >= 2.4)
                continue;

            TightLepton_pt.push_back(Muon_pt[iMuon]);
            TightLepton_eta.push_back(Muon_eta[iMuon]);
            TightLepton_phi.push_back(Muon_phi[iMuon]);

            // Count the number of tight leptons
            nTightLepton++;
        }

        // First event level selection
        if (not (nVetoLepton == 1 and nTightLepton == 1))
            continue;

        for (unsigned int iFatJet = 0; iFatJet < nFatJet; ++iFatJet)
        {
            if (FatJet_pt[iFatJet] <= 250)
                continue;
            if (FatJet_mass[iFatJet] <= 50)
                continue;
            if (FatJet_msoftdrop[iFatJet] <= 40)
                continue;

            // Compute the distance in eta v. phi space
            float dr = DeltaR(TightLepton_eta[0], TightLepton_phi[0], FatJet_eta[iFatJet], FatJet_phi[iFatJet]);
            bool is_overlap = dr < 0.8; // if the distance is less than 0.8, it is considered overlapping

            // if overlapping skip
            if (is_overlap)
                continue;

            nGoodFatJet++;

            // Compute the Xbb score (this is a score for a machine learning output classifying Higgs boson from non-Higgs boson initiated fatjets)
            float xbb_score = FatJet_particleNetMD_Xbb[iFatJet] / (FatJet_particleNetMD_Xbb[iFatJet] + FatJet_particleNetMD_QCD[iFatJet]);

            // Keep track of the fatjet with the highest Xbb score
            if (xbb_score > MaxHbbScore)
            {
                MaxHbbScore = xbb_score;
                HbbFatJet_pt = FatJet_pt[iFatJet];
            }

        }

        // Second event level selection
        if (nGoodFatJet < 1)
            continue;


        // Third event level selection Require ST > 800 GeV
        float ST = HbbFatJet_pt + TightLepton_pt[0] + MET_pt;
        if (ST < 800)
            continue;

        // Selection for Jets
        for (unsigned int iJet = 0; iJet < nJet; ++iJet)
        {
            // each jet must have pt greater than 20
            if (Jet_pt[iJet] <= 20)
                continue;

            // Compute the distance in eta v. phi space
            float dr = DeltaR(TightLepton_eta[0], TightLepton_phi[0], Jet_eta[iJet], Jet_phi[iJet]);
            bool is_overlap = dr < 0.4; // if the distance is less than 0.4, it is considered overlapping

            // if overlapping skip
            if (is_overlap)
                continue;

            // Count the number of good jets
            nGoodJet++;
        }

        // Fourth event level selection
        if (nGoodJet < 2)
            continue;

        
        //std::cout <<  "putting ientry: " << ientry <<  std::endl;

        // With the following function call, the "column" variables are filled
        tree->GetEntry(ientry);

        // Once we reach this point we fill the event to the output ttree if only if it reaches here
        finalevents++;
        outputTree->Fill();

    }       

    auto stop_f = std::chrono::system_clock::now();
    auto total_f = std::chrono::duration_cast<std::chrono::milliseconds>(stop_f - start_f);


    std::cout  <<  " Filter and fill tree" <<  total_f.count() << " milliseconds\n";


    auto start_w = std::chrono::system_clock::now();

    outputFile.Write();

    auto stop_w = std::chrono::system_clock::now();
    auto total_w = std::chrono::duration_cast<std::chrono::milliseconds>(stop_w - start_w);


    std::cout  <<  " Time w/o write to buffer" <<  total_w.count() << " milliseconds\n";

    auto start_b = std::chrono::system_clock::now();
    // Use a TBufferFile for serialization of the outputTree
    TBufferFile buf(TBuffer::kWrite);

    // Serialize the outputTree into the buffer
    buf.WriteObject(outputTree);

    // Access the internal buffer of TBufferFile
    Int_t bufSize = buf.Length();
    void* newBuffer = malloc(bufSize);  // Allocate a new buffer
    if (!newBuffer) {
        std::cerr << "Memory allocation failed" << std::endl;
        return nullptr;
    }
    memcpy(newBuffer, buf.Buffer(), bufSize);  // Copy the content to the new buffer

    // Update the reference to the size of the new buffer
    filtered_size = static_cast<size_t>(bufSize);

    auto stop_b = std::chrono::system_clock::now();
    auto total_b = std::chrono::duration_cast<std::chrono::milliseconds>(stop_b - start_b);


    std::cout  <<  "  write to buffer" <<  total_b.count() << " milliseconds\n";

    auto stop = std::chrono::system_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);


    std::cout  <<  " Total time " <<  total.count() << " milliseconds and read " << totread << "bytes\n";

    // The newBuffer now contains the serialized data of outputTree
    // It should be returned or used as needed
    // Remember: It's the caller's responsibility to free this memory
    return reinterpret_cast<unsigned char*>(newBuffer);

    }

}