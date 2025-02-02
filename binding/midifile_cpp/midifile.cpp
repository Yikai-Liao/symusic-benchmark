#include "MidiFile.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using namespace nb::literals;

smf::MidiFile load(std::string file){
    smf::MidiFile midifile;
    midifile.read(file);
    midifile.doTimeAnalysis();
    midifile.linkNotePairs();
    return midifile;
}

NB_MODULE(midifile_cpp, m) {
    m.doc() = "A simple binding of midifile for benchmark in python.";
    nb::class_<smf::MidiFile>(m, "MidiFile")
        .def("dump_midi", [](smf::MidiFile &midifile, const std::string& path){
            midifile.write(path);
        }, "Dump the midi file to the given path.")
    ;

    m.def("load", &load, "Load a midi file from the given path.");

    m.def("dump", [](smf::MidiFile &midifile, const std::string& path){
        midifile.write(path);
    }, "Dump the midi file to the given path.");

    // version
    m.attr("__version__") = "0.0.1";
}