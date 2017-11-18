/*
 * MIT License
 *
 * Copyright (c) 2014-2018 David Moskowitz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.infoblazer.gp.evolution.model;

import com.infoblazer.gp.application.data.model.Metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by David on 5/25/2015.
 */
public class Population {

    private List<ResultProducingProgram> resultPopulation;
    private List<RegimeDetectionProgram> regimePopulation;
    private Metrics metrics;
    private Metrics regimeMetrics;


    public Population(Winners winners) {
        resultPopulation = new ArrayList<>();
        resultPopulation.add(winners.getResultProducingProgram());
        regimePopulation = new ArrayList<>();
        regimePopulation.add(winners.getRegimeDetectionProgram());

    }
    public Population() {
    }

    public Population(List<ResultProducingProgram> resultPopulation,List<RegimeDetectionProgram> regimePopulation) {
        this.resultPopulation = resultPopulation;
        this.regimePopulation = regimePopulation;
    }
    public Population(ResultProducingProgram resultProducingProgram, RegimeDetectionProgram regimeDetectionProgram) {
        resultPopulation = new ArrayList<>();
        resultPopulation.add(resultProducingProgram);
        regimePopulation = new ArrayList<>();
        regimePopulation.add(regimeDetectionProgram);
    }

    public Integer getRPLength(){
        return resultPopulation.size();
    }
    public Integer getRGLength(){
        if (regimePopulation==null){
            return  0;
        }else {
            return regimePopulation.size();
        }
    }

    public List<ResultProducingProgram> getResultPopulation() {
        return this.resultPopulation;
    }

    public void setResultPopulation(List<ResultProducingProgram> resultPopulation) {
        this.resultPopulation = resultPopulation;
    }

    public List<RegimeDetectionProgram> getRegimePopulation() {
        return this.regimePopulation;
    }

    public void setRegimePopulation(List<RegimeDetectionProgram> regimePopulation) {
        this.regimePopulation = regimePopulation;
    }

    public void setResultPopulation(ResultProducingProgram[] resultProducingPrograms) {
        resultPopulation = Arrays.asList(resultProducingPrograms);
    }

    public void setRegimePopulation(RegimeDetectionProgram[] regimeDetectionPrograms) {
        if (regimeDetectionPrograms==null){
            regimePopulation = null;
        }else {
            regimePopulation = Arrays.asList(regimeDetectionPrograms);
        }
    }

    public Metrics getRegimeMetrics() {
        return this.regimeMetrics;
    }

    public void setRegimeMetrics(Metrics regimeMetrics) {
        this.regimeMetrics = regimeMetrics;
    }

    public Metrics getMetrics() {
        return this.metrics;
    }

    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

}
