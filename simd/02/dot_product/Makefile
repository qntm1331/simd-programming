CXX=g++
CXXFLAGS=-mavx2 -mfma -masm=att -std=c++11
TARGET=simd_program
BUILDDIR=build
ASMFILE=$(BUILDDIR)/main.s
SRCFILE=main.cpp
OBJFILE=$(BUILDDIR)/main.o
EXECUTABLE=$(BUILDDIR)/$(TARGET)

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRCFILE) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(SRCFILE) -o $(EXECUTABLE)

asm: $(SRCFILE) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -S $(SRCFILE) -o $(ASMFILE)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(TARGET)

