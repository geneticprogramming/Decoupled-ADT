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

package com.infoblazer.gp.evolution.selectionstrategy;

import com.infoblazer.gp.application.data.model.FitnessEvaluation;
import com.infoblazer.gp.application.fitness.FitnessEvaluator;
import com.infoblazer.gp.evolution.model.*;
import com.infoblazer.gp.evolution.primitives.FunctionSet;
import com.infoblazer.gp.evolution.primitives.GP_TYPES;
import com.infoblazer.gp.evolution.primitives.Primitive;
import com.infoblazer.gp.evolution.primitives.TerminalSet;
import com.infoblazer.gp.evolution.primitives.functions.BinaryNumber;
import com.infoblazer.gp.evolution.primitives.functions.Function;
import com.infoblazer.gp.evolution.utils.GpUtils;
import org.apache.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by David on 8/7/2014.
 */
public abstract class AbstractSelectionStrategy implements SelectionStrategy {
    public static final int CROSSOVER_ATTEMPTS = 5;
    protected Random random = new Random();
    @Autowired
    protected Compression compression;
    @Autowired
    protected Expansion expansion;
    @Autowired
    protected CrossOver crossOver;
    @Autowired
    protected Mutation mutation;
    private final static Logger logger = Logger.getLogger(AbstractSelectionStrategy.class.getName());

    protected FitnessEvaluator fitnessEvaluator;
    @Value("${regimeSelection:#{true}}")
    protected Boolean regimeSelection;  // always use reproduction on regimes
    @Value("${fourWayPct:#{50}}")
    protected Double fourWayPct;   //change to use a tournament to select other program to use in fitness calculation (ex. regime program to calculate result)
    @Value("${crossoverPct}")
    protected Double crossoverPct;
    @Value("${mutationPct}")
    protected double mutationPct;
    @Value("${compressionPct:#{0.0}}")
    protected Double compressionPct;
    @Value("${tournamentSize:#{2}}")
    protected int tournamentSize;
    @Value("${expansionPct:#{0.0}}")
    protected Double expansionPct;
    protected FunctionSet functionSet;
    protected FunctionSet regimeFunctionSet;
    protected TerminalSet terminalSet;
    protected List<String> series;
    protected String[] adfArities;
    @Value("${maxInitDepth}")
    protected int maxInitDepth;

    @Value("${maxDepth:#{999999}}")  //essentially unlimited if not set
    protected int maxDepth;

    @Value("${maxSize:#{999999}}")  //essentially unlimited if not set
    protected int maxSize;
    @Value("${elitist:#{false}}")
    protected Boolean elitist;

    @Value("${regimes:#{1}}")
    protected int regimes;

    @Value("${useAverageFitnessSelector:#{false}}")
    private Boolean useAverageFitnessSelector;  //usually default of false is used, take best fitness


    public void setFitnessEvaluator(FitnessEvaluator fitnessEvaluator) {
        this.fitnessEvaluator = fitnessEvaluator;
    }

    public List<String> getSeries() {
        return series;
    }

    public void setSeries(List<String> series) {
        this.series = series;
    }

    protected Direction direction;

    @Override
    public void setDirection(Direction direction) {
        this.direction = direction;
    }

    @Override
    public Direction getDirection() {
        return direction;
    }


    public void setRegimeFunctionSet(FunctionSet regimeFunctionSet) {
        this.regimeFunctionSet = regimeFunctionSet;
    }

    public void setFunctionSet(FunctionSet functionSet) {
        this.functionSet = functionSet;
    }

    public void setTerminalSet(TerminalSet terminalSet) {
        this.terminalSet = terminalSet;
    }


    public static void addFunctions(List<Primitive> primitives, Primitive root) {
        if (root != null) {   //null is sometimes past in
            if (root instanceof Function) {
                primitives.add(root);
                Function function = (Function) root;
                for (Primitive primitive : function.getParameters()) {
                    addFunctions(primitives, primitive);
                }
            }
        }
    }


    @Override
    public Population selectionNextGeneration(int generation, int trainingGenerations, GrowMethod growMethod,
                                              Population population, Integer maxTotalNodes, Integer windowStart,
                                              Integer windowEnd, Integer predictedRegime, boolean lastPredictionTrainingRound) {

        List<ResultProducingProgram> nextGenerationRP = new ArrayList<>();
        List<RegimeDetectionProgram> nextGenerationRG = new ArrayList<>();
        Integer targetRpPopulationSize = null;
        Integer targetRGPopulationSize = null;

        if (maxTotalNodes == null) {
            targetRpPopulationSize = population.getRPLength();
        }
        if (regimes == 1) {
            targetRGPopulationSize = 0;
        } else if (maxTotalNodes == null) {
            targetRGPopulationSize = population.getRGLength();
        }


        ResultProducingProgram fittestResultProducingProgram = (ResultProducingProgram) ResultProducingProgram.findFittest(population.getResultPopulation(), direction);
        RegimeDetectionProgram fittestRegimeDetectionProgram = null;
        if (regimes > 1) {
            fittestRegimeDetectionProgram = (RegimeDetectionProgram) RegimeDetectionProgram.findFittest(population.getRegimePopulation(), direction);
        }
        if (elitist) {
            ResultProducingProgram eliteRP = (ResultProducingProgram) GpUtils.getKyroInstance().copy(fittestResultProducingProgram);
            RegimeDetectionProgram eliteRegime = null;
            if (fittestRegimeDetectionProgram != null) {
                eliteRegime = (RegimeDetectionProgram) GpUtils.getKyroInstance().copy(fittestRegimeDetectionProgram);
            }


            FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(windowStart, windowEnd, maxDepth, eliteRP, eliteRegime, direction);
            eliteRP.setFitness(fitnessEvaluation.getFitness());
            eliteRP.calculateMetrics();
            nextGenerationRP.add(eliteRP);
            if (eliteRegime != null) {
                eliteRegime.calculateMetrics();
                eliteRegime.setFitness(fitnessEvaluation.getFitness());
                nextGenerationRG.add(eliteRegime);
            }
        }


        //initially add the same number of individuals as the last generation. Then resize population
        Boolean nodeLimitReached = false;
        int totalRPNodes = 0;
        int totalRGNodes = 0;
        int nextResultId = 1;//need to set this before fitness calculation, dyfor uses this.
        int nextRegimeId = 1;
        while (needPopulation(maxTotalNodes, nextGenerationRP, nextGenerationRG,
                targetRpPopulationSize, targetRGPopulationSize, nodeLimitReached)) {

            Population newChildren = chooseChildren(growMethod, population, predictedRegime);

            if (newChildren != null) {
                if (newChildren.getResultPopulation() != null) {
                    for (ResultProducingProgram child : newChildren.getResultPopulation()) {
                        if (targetRpPopulationSize == null || nextGenerationRP.size() < targetRpPopulationSize) {
                            child.setId(nextResultId); //This is needed here for dyfor GP, id is used to determine program window to use for fitness calcualtion

                            RegimeDetectionProgram regimeDetectionProgram = fittestRegimeDetectionProgram;
                            if (regimes > 1) {
                                double rnd = random.nextDouble()*100;
                                if (rnd < fourWayPct)
                                    regimeDetectionProgram = (RegimeDetectionProgram) runTournament(population.getRegimePopulation(), tournamentSize);
                            }

                            FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(windowStart, windowEnd, maxDepth,
                                    child, regimeDetectionProgram, direction); //was
                            child.setFitness(fitnessEvaluation.getFitness());
                            child.calculateMetrics();
                            child.calculateAdfMetrics(regimes);

                            totalRPNodes += child.getNodeCount() + null2Zero(child.getTotalAdfNodeCount());
                            nodeLimitReached = checkNodeLimit(maxTotalNodes, totalRPNodes + totalRGNodes);
                            if (child.getNodeCount() > maxSize || child.getDepth() > maxDepth) {
                                child.setFitness(direction.getMinFitness());
                            } else if (child.getMaxAdfNodeCount() != null &&
                                    (child.getMaxAdfNodeCount() > maxSize || child.getMaxAdfDepth() > maxDepth)) {
                                child.setFitness(direction.getMinFitness());
                            } else {
                                nextGenerationRP.add(child);
                                nextResultId++;
                            }

                        }
                    }
                }
                if (newChildren.getRegimePopulation() != null) {
                    for (RegimeDetectionProgram child : newChildren.getRegimePopulation()) {
                        if (child != null) {
                            if (targetRGPopulationSize == null || nextGenerationRG.size() < targetRGPopulationSize) {
                                child.setId(nextRegimeId);
                                double rnd = random.nextDouble()*100;
                                ResultProducingProgram resultProducingProgram = null;
                                if (rnd < fourWayPct) {
                                    resultProducingProgram = (ResultProducingProgram) runTournament(population.getResultPopulation(), tournamentSize);
                                } else {
                                    resultProducingProgram = fittestResultProducingProgram;
                                }
                                FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(windowStart, windowEnd, maxDepth,
                                        resultProducingProgram, child, direction);
                                child.setFitness(fitnessEvaluation.getFitness());
                                child.calculateMetrics();
                                child.calculateAdfMetrics(regimes);

                                totalRGNodes += child.getNodeCount() + null2Zero(child.getTotalAdfNodeCount());
                                nodeLimitReached = checkNodeLimit(maxTotalNodes, totalRPNodes + totalRGNodes);
                                if (child.getNodeCount() > maxSize || child.getDepth() > maxDepth) {
                                    child.setFitness(direction.getMinFitness());
                                } else if (child.getMaxAdfNodeCount() != null &&
                                        (child.getMaxAdfNodeCount() > maxSize || child.getMaxAdfDepth() > maxDepth)) {
                                    child.setFitness(direction.getMinFitness());
                                } else {
                                    nextGenerationRG.add(child);
                                    nextRegimeId++;
                                }
                            }
                        }
                    }
                }
            }

        }

        Population newPopulation = new Population(nextGenerationRP, nextGenerationRG);

        fitnessEvaluator.afterGeneration(newPopulation, direction, generation, trainingGenerations, maxDepth, windowEnd, lastPredictionTrainingRound);
        return newPopulation;
    }

    abstract protected AbstractProgram runTournament(List<? extends AbstractProgram> programs, int participantCount);

    private Integer null2Zero(Integer val) {
        return val == null ? 0 : val;
    }

    private boolean needPopulation(Integer maxTotalNodes, List<ResultProducingProgram> nextGenerationRP, List<RegimeDetectionProgram> nextGenerationRG, Integer targetRpPopulationSize, Integer targetRGPopulationSize, Boolean nodeLimitReached) {
        boolean result = true;
        if (maxTotalNodes != null && nodeLimitReached) {
            result = false;
        } else if (maxTotalNodes == null &&
                (nextGenerationRP.size() >= targetRpPopulationSize &&
                        nextGenerationRG.size() >= targetRGPopulationSize)) {
            result = false;
        }
        return result;
    }

    private Boolean checkNodeLimit(Integer maxTotalNodes, int nodes) {
        return maxTotalNodes != null && maxTotalNodes < nodes;
    }

    private Population chooseChildren(
            GrowMethod growMethod,
            Population population,
            Integer predictedRegime
    ) {
        Population result = null;

        //can return one or two
        //do tournament selection

        double crossoverBracket = crossoverPct / 100d;
        double mutationBracket = (mutationPct / 100d) + crossoverBracket;
        double compressBracket = (compressionPct / 100d) + mutationBracket;
        double expandBracket = (expansionPct / 100d) + compressBracket;
        Double selectionRoll = random.nextDouble();
        if (selectionRoll <= crossoverBracket) { //crossover
            logger.debug("selected Crossover");
            result = crossOver(population, predictedRegime);
        } else if (selectionRoll <= mutationBracket) {
            logger.debug("selected mutation");

            Winners winners = selectWinners(population);
            result = new Population();
            ResultProducingProgram resultProducingProgram = (ResultProducingProgram) mutation(winners.getResultProducingProgram(), growMethod, false, predictedRegime);
            RegimeDetectionProgram regimeDetectionProgram = winners.getRegimeDetectionProgram();
            if (regimes > 1) {
                regimeDetectionProgram = (RegimeDetectionProgram) mutation(winners.getRegimeDetectionProgram(), growMethod, true, predictedRegime);
            }

            result.setResultPopulation(Arrays.asList(resultProducingProgram));
            result.setRegimePopulation(Arrays.asList(regimeDetectionProgram));
        } else if (selectionRoll <= compressBracket) {
            logger.debug("selected compression");
            //Note this is basically a duplicate of the mutation. may be inherit or something

            Winners winners = selectWinners(population);
            result = compression.compress(winners, regimes, maxDepth);
        } else if (selectionRoll <= expandBracket) {
            logger.debug("selected expansion");
            Winners winners = selectWinners(population);
            result = expansion.expand(winners, regimes, maxDepth,  predictedRegime);
        } else {//reproduction
            logger.debug("selected reproduction");
            Winners winners = selectWinners(population);
            result = new Population(winners);

        }

        return result;
    }


    protected abstract Winners selectWinners(Population population);

    @Override
    public Pair<AbstractProgram> doCrossOver(AbstractProgram parent1, AbstractProgram parent2,
                                             int maxDepth, int maxSize, Direction direction, int regimes, boolean isRegimeDetection, Integer predictedRegime) {
        return crossOver.doCrossOver(parent1, parent2, maxDepth, maxSize, direction, regimes, isRegimeDetection, predictedRegime);
    }

    public static void addPrimitivesTyped(List<Primitive> primitives, Primitive root, Class clazz) {
        if (root != null) { //null is sometimes past in
            if (root.getClass().getName().equals(clazz.getName())) {
                primitives.add(root);
            }
            if (root instanceof Function) {
                Function function = (Function) root;
                for (Primitive primitive : function.getParameters()) {
                    addPrimitivesTyped(primitives, primitive, clazz);
                }

            }
        }
    }


    private Population crossOver(Population population, Integer predictedRegime) {

        Population children = new Population();
        logger.trace("crossover selected");
        ResultProducingProgram[] parents = new ResultProducingProgram[2];
        RegimeDetectionProgram[] regimeParents = new RegimeDetectionProgram[2];
        int i = 0;

        while (i < 2) {
            Winners winner = selectWinners(population);
            parents[i] = winner.getResultProducingProgram();
            regimeParents[i] = winner.getRegimeDetectionProgram();
            i++;
        }


        //do crossover, but return p1 for now

        Pair<? extends AbstractProgram> resultProducingChildren = null;
        Pair<? extends AbstractProgram> regimeDetectingChildren = null;
        int counter = 0;
        while (resultProducingChildren == null && counter < CROSSOVER_ATTEMPTS) {
            counter++;
            resultProducingChildren = crossOver.doCrossOver(parents[0], parents[1], maxDepth, maxSize, direction, regimes, false, predictedRegime);
        }
        children.setResultPopulation(resultProducingChildren.asResultProducingList());
        if (regimes > 1) {
            while (regimeDetectingChildren == null && counter < CROSSOVER_ATTEMPTS) {
                counter++;
                if (regimeSelection) {
                    regimeDetectingChildren = crossOver.doCrossOver(regimeParents[0], regimeParents[1], maxDepth, maxSize, direction, regimes, true, predictedRegime);
                } else {
                    AbstractProgram copy1 = (AbstractProgram) GpUtils.getKyroInstance().copy(regimeParents[0]);
                    AbstractProgram copy2 = (AbstractProgram) GpUtils.getKyroInstance().copy(regimeParents[1]);
                    regimeDetectingChildren = new Pair<>(copy1, copy2);
                }
            }
            children.setRegimePopulation(regimeDetectingChildren.asRegimeDetectionList());
        }

        return children;
    }

    @Override
    public AbstractProgram mutation(AbstractProgram program, GrowMethod growMethod, boolean isRegimeDetection, Integer predictedRegime) {
        FunctionSet effectiveFunctionSet = null;
        if (isRegimeDetection) {
            effectiveFunctionSet = regimeFunctionSet;
        } else {
            effectiveFunctionSet = functionSet;
        }
        return mutation.mutation(program, effectiveFunctionSet, terminalSet, series, growMethod, maxInitDepth, maxDepth, isRegimeDetection, predictedRegime);
    }

    /**
     * @param primitives
     * @param root
     * @param level
     * @param maxLevel
     * @return number of levels found
     * @
     */
    public static int addPrimitives(List<Primitive> primitives, Primitive root, final Integer level) {
        int thisLevel;
        Integer startLevel = level;
        if (level == null) {
            startLevel = 1;
        }
        thisLevel = startLevel;


        if (root != null) { //null is sometimes past in
            primitives.add(root);
            if (root instanceof Function) {
                Function function = (Function) root;
                for (Primitive primitive : function.getParameters()) {
                    int nextLevel = addPrimitives(primitives, primitive, startLevel + 1);
                    if (nextLevel > thisLevel) {
                        thisLevel = nextLevel;
                    }
                }

            }
        }
        return thisLevel;
    }


    public static boolean checkSize(int maxDepth, int maxSize, int nodeDepth, int nodeSize) {
        Boolean result = (nodeDepth <= maxDepth && nodeSize <= maxSize);
        return result;
    }

    public static Primitive getStrongType(List<Primitive> primitives, GP_TYPES returnType) {
        Random random = new Random();
        Primitive primitive = null;
        if (primitives.size() > 0) {
            int counter = 0;
            while (primitive == null && counter++ < 100) { //failsafe
                int selectionPoint = random.nextInt(primitives.size());
                Primitive potentialPrimitive = primitives.get(selectionPoint);

                if (potentialPrimitive != null && !(potentialPrimitive instanceof BinaryNumber)) {
                    if (returnType == potentialPrimitive.getReturnType()) {
                        primitive = potentialPrimitive;
                    }
                }
            }
        }
        return primitive;
    }


}
